#!/usr/bin/env python3
"""
Self-contained driver for TASS (Temperature-Accelerated Sliced Sampling)
reweighting and free-energy computation across multiple umbrella windows.

This is a single-file program: the ``plumed.dat`` parser and the ``c(t)`` /
``V_bias`` reweighting kernels are all included here.  The only required
third-party dependency is NumPy (optional: ``mpi4py`` for parallel ``vbias``,
``numba`` for the threaded kernel).

The simulation parameters are read directly from each window's ``plumed.dat``
(the parser is included in this module):

  * ncv, iwrap (TORSION -> 1), icv_metad (CVs carrying a metadynamics bias),
    icv_restraint (CV carrying the umbrella RESTRAINT)
  * KAPPA / AT of the RESTRAINT, PACE (w_hill) and BIASFACTOR of the METAD,
    the TEMP of the EXTENDED_LAGRANGIAN line, and the COLVAR print STRIDE (w_cv)

``ncv``, ``iwrap``, ``icv_restraint`` and the restraint CV index ``cv1`` may all
differ from window to window (each window's ``plumed.dat`` defines its own CV
layout).  Within a window, ``cv1`` is the 0-based index of the CV carrying the
umbrella ``RESTRAINT`` (the entry of ``icv_restraint`` equal to 1).  ``icv_metad``
may also differ: each window can use a different set of metadynamics CVs, or none.
action needs no HILLS, no ct.dat and no vbias.dat -- its reweighting weight is
simply 1 for every frame (equivalent to V^bias = 0 and c(t) = 0).

This program computes, for every window:
  1. c(t) factor                ->  ct.dat     (serial, NumPy)
  2. V^bias(s,t) bias potential ->  vbias.dat  (PARALLEL: MPI + threads)
  3. the reweighted mean force   dF1/ds1  at the window's restraint centre

and then, globally:
  4. F1(s1) along the restraint CV by trapezoidal integration of the mean
     force over the windows                  ->  fes_1d.dat, mean_force.dat
  5. (optional, if PRINT2D) the 2D free energy F(cv1, cv2) by WT-MetaD
     reweighting, stitched with F1(s1)        ->  free_energy_2D.dat

The reweighted mean force at each window is

    dF1/ds1 = < -kappa_restraint * (s1(t) - s1_at) >_reweighted

with the reweighting weight w(t) = exp( (V^bias(t) - c(t)) / kT ).

Periodic boundary conditions (period 2*pi) are applied to any difference of 
TORSION-type CVs.

tass.inp format
---------------
    <nwin>                          # number of windows
    # then, for each window:
    <folder>                        # folder with plumed.dat, COLVAR, HILLS
    GRIDS_METAD | AUTO_GRIDS_METAD
      # if GRIDS_METAD: one line per metad CV:  icv_mtd  grid_min  grid_max  dgrid
    <t_min> <t_max>                 # t_max < 0  ->  use all COLVAR frames
    # after all windows, optionally:
    PRINT2D [AUTO_GRID]
      <icv2> [grid_min grid_max dgrid]   # 2nd CV for the 2D surface (not the
                                         # restraint CV); grids auto if AUTO_GRID

Usage:
    python tass_reweight.py [tass.inp] [--root DIR]
                            [--nranks N] [--threads T] [--backend numpy|numba]
                            [--force] [-v]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()   # used to re-exec self for the vbias worker

# Physical / numerical constants / Settings
KB = 8.314472e-3        # kJ/(mol*K)
MIN_PROB = 1.0e-32      # probability floor inside the log
TWOPI = 2.0 * np.pi     # 2 Pi
AUTO_NBINS = 50         # auto grids: 50 bins -> 51 points


# =========================================================================== #
#  Generic helpers and reweighting kernels
# =========================================================================== #
def _to_float(token: str) -> float:
    """Parse a Fortran-style real such as '-3.2d0' or '1.0E-3'."""
    return float(token.replace("d", "e").replace("D", "e"))


def _nint(x):
    """Fortran NINT: round half away from zero (numpy rounds half to even)."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def count_lines(path: Path) -> int:
    """Count data lines, skipping blank lines and '#' comments/headers."""
    n = 0
    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            n += 1
    return n


# Reads a numeric data file (like COLVAR, vbias.dat, or ct.dat) into a NumPy 2D array, 
# of size (n_rows, n_columns) with strict checks and PLUMED-style comment handling.
#
def load_table(path: Path, label: str, min_cols: int) -> np.ndarray:
    """Load a whitespace-separated numeric file, skipping '#'/blank lines."""
    path = Path(path)
    if not path.is_file():
        sys.exit(f"!!ERROR: file not found while reading {label}: {path}")
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            fields = s.split()
            if len(fields) < min_cols:
                sys.exit(
                    f"!!ERROR reading {label} ({path}) line {lineno}: "
                    f"expected at least {min_cols} columns, got {len(fields)}: {s!r}"
                )
            try:
                rows.append([_to_float(x) for x in fields])
            except ValueError:
                sys.exit(
                    f"!!ERROR reading {label} ({path}) line {lineno}: "
                    f"cannot parse as numbers: {s!r}"
                )
    if not rows:
        sys.exit(f"!!ERROR: no data found in {label} ({path})")
    width = max(len(r) for r in rows)
    #Pads shorter rows with NaN if some lines have fewer columns than the widest row.
    if any(len(r) != width for r in rows):
        rows = [r + [np.nan] * (width - len(r)) for r in rows]
    return np.asarray(rows, dtype=np.float64)


class InputReader:
    """Line-oriented reader for ``tass.inp``.

    Each logical READ consumes one record (line); trailing '#...' comments and
    extra tokens are ignored, and blank lines are skipped.
    """

    def __init__(self, text: str):
        self._lines = text.splitlines()
        self._i = 0

    #Internal: returns the next non-blank line, or None at EOF
    def _advance(self) -> str | None:
        while self._i < len(self._lines):
            raw = self._lines[self._i]
            self._i += 1
            if raw.strip() == "":
                continue
            return raw
        return None

    #Next line -> list of strings; exits with error if EOF
    def tokens(self, label: str) -> list[str]:
        raw = self._advance()
        if raw is None:
            sys.exit(f"!!ERROR: unexpected end of input while reading {label}")
        return raw.split("#", 1)[0].split()

    #Next line as a string (no splitting); used to detect PRINT2D
    def raw_line(self) -> str | None:
        return self._advance()

    #Next line -> first token as int
    def one_int(self, label: str) -> int:
        return int(self.tokens(label)[0])

    #Next line -> first n tokens as integers
    def ints(self, n: int, label: str) -> list[int]:
        tok = self.tokens(label)
        if len(tok) < n:
            sys.exit(f"!!ERROR: expected {n} integers for {label}, got {tok}")
        return [int(t) for t in tok[:n]]

    #Next line -> first n tokens as floats (supports 1.0d0)
    def floats(self, n: int, label: str) -> list[float]:
        tok = self.tokens(label)
        if len(tok) < n:
            sys.exit(f"!!ERROR: expected {n} reals for {label}, got {tok}")
        return [_to_float(t) for t in tok[:n]]


def nbins_from_grid(gmin: float, gmax: float, dgrid: float) -> int:
    """Grid spacing -> number of bins."""
    return int(round((gmax - gmin) / dgrid)) + 1


def ct_read_hills(path: Path, mtd_steps: int, ndim: int):
    """HILLS -> (hill, width, ht) over ``ndim`` metad CVs (1+2*ndim+1 columns)."""
    nd = int(ndim)
    cols = 1 + 2 * nd + 1
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[0] != mtd_steps:
        raise ValueError(f"HILLS rows ({data.shape[0]}) != mtd_steps ({mtd_steps})")
    if data.shape[1] < cols:
        raise ValueError(f"HILLS columns ({data.shape[1]}) < expected {cols}")
    hill = data[:, 1:1 + nd].T.copy()
    width = data[:, 1 + nd:1 + 2 * nd].T.copy()
    ht = data[:, 1 + 2 * nd].copy()
    return hill, width, ht


def build_grid_coords(gridmin, gridmax, nbin):
    """Coordinates of all grid bins, shape (n_total, ndim)."""
    axes = [np.linspace(gridmin[d], gridmax[d], int(nbin[d]), dtype=np.float64)
            for d in range(gridmin.size)]
    grids = np.meshgrid(*axes, indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=1)


def wrap_diff(diff, mask, twopi=TWOPI):
    """diff - twopi*nint(diff/twopi) applied only to periodic dims."""
    wrapped = diff - twopi * np.rint(diff / twopi)
    return np.where(mask, wrapped, diff)


def compute_ct(hill, width, ht, coords, kt_energy, gamma_, iwrap, twopi=TWOPI):
    """c(t) factor for WT-MetaD (arbitrary metad-CV dimension)."""
    _, mtd_steps = hill.shape
    n_total = coords.shape[0]
    mask = np.asarray(iwrap).reshape(-1).astype(bool)
    any_periodic = bool(mask.any())

    fes = np.zeros(n_total, dtype=np.float64)
    ct = np.empty(mtd_steps, dtype=np.float64)
    for i in range(mtd_steps):
        diff = coords - hill[:, i]
        if any_periodic:
            diff = wrap_diff(diff, mask, twopi)
        w2 = width[:, i] * width[:, i]
        diff2_sum = 0.5 * np.sum((diff * diff) / w2, axis=1)
        fes -= float(ht[i]) * np.exp(-diff2_sum)
        num = float(np.sum(np.exp(-fes / kt_energy)))
        den = float(np.sum(np.exp(-fes / kt_energy + fes * gamma_ / kt_energy)))
        if den <= 0.0 or num <= 0.0:
            raise RuntimeError(f"non-positive accumulator at MTD step {i+1}")
        ct[i] = kt_energy * np.log(num / den)
    return ct


def vbias_load_colvar(path: Path, metad_idx):
    """COLVAR -> cv (md_steps, ncv_mtd): only the metad-CV columns (col d+1)."""
    cols = tuple(int(d) + 1 for d in metad_idx)
    cv = np.loadtxt(path, comments="#", usecols=cols, ndmin=2)
    return np.ascontiguousarray(cv, dtype=np.float64)


def vbias_load_hills(path: Path, ncv_mtd: int):
    """HILLS -> (centers, widths, heights) over ncv_mtd metad CVs."""
    c_cols = tuple(range(1, ncv_mtd + 1))
    s_cols = tuple(range(ncv_mtd + 1, 2 * ncv_mtd + 1))
    h_col = (2 * ncv_mtd + 1,)
    centers = np.loadtxt(path, comments="#", usecols=c_cols, ndmin=2)
    widths = np.loadtxt(path, comments="#", usecols=s_cols, ndmin=2)
    heights = np.loadtxt(path, comments="#", usecols=h_col, ndmin=2).ravel()
    return (np.ascontiguousarray(centers, dtype=np.float64),
            np.ascontiguousarray(widths, dtype=np.float64),
            np.ascontiguousarray(heights, dtype=np.float64))


def vbias_steps_numpy(steps, cv, centers, inv_sig2, ht_gamma, iwrap, w_cv, w_hill):
    """Vectorised NumPy kernel: V_bias for each MD step in ``steps`` (1-based)."""
    out = np.empty(steps.size, dtype=np.float64)
    wrap_mask = iwrap.astype(bool)
    for k in range(steps.size):
        i_md = steps[k]
        mtd_max = (i_md * w_cv) // w_hill
        if mtd_max <= 0:
            out[k] = 0.0
            continue
        diff = cv[i_md - 1] - centers[:mtd_max]
        if wrap_mask.any():
            d = diff[:, wrap_mask]
            diff[:, wrap_mask] = d - TWOPI * np.round(d / TWOPI)
        arg = 0.5 * np.einsum("md,md->m", diff * diff, inv_sig2[:mtd_max])
        out[k] = np.dot(ht_gamma[:mtd_max], np.exp(-arg))
    return out


def _build_numba_kernel():
    """Compile the threaded (OpenMP) numba kernel lazily; None if unavailable."""
    try:
        from numba import njit, prange
    except Exception:
        return None

    @njit(parallel=True, fastmath=True, cache=True)
    def _kernel(steps, cv, centers, inv_sig2, ht_gamma, iwrap, w_cv, w_hill):
        n = steps.shape[0]
        ncv = cv.shape[1]
        out = np.zeros(n, dtype=np.float64)
        for k in prange(n):
            i_md = steps[k]
            mtd_max = (i_md * w_cv) // w_hill
            acc = 0.0
            for m in range(mtd_max):
                arg = 0.0
                for d in range(ncv):
                    diff = cv[i_md - 1, d] - centers[m, d]
                    if iwrap[d] == 1:
                        diff -= TWOPI * np.round(diff / TWOPI)
                    arg += 0.5 * diff * diff * inv_sig2[m, d]
                acc += ht_gamma[m] * np.exp(-arg)
            out[k] = acc
        return out

    return _kernel


# =========================================================================== #
#  plumed.dat parser
# =========================================================================== #
@dataclass
class MetadInfo:
    """Metadynamics bias acting on one or more CVs."""

    label: str
    cv_indices: List[int]
    cv_names: List[str]
    w_hill: int                      # value of PACE
    biasfactor: Optional[float]
    temp: Optional[float]            # value of TEMP (optional; warned if absent)
    height: Optional[float] = None
    sigma: Optional[List[float]] = None


@dataclass
class RestraintInfo:
    """A RESTRAINT acting on one CV."""

    label: str
    cv_index: int
    cv_name: str
    kappa_restraint: float
    restraint_at: float


@dataclass
class PlumedInput:
    """Structured result of parsing a ``plumed.dat`` file."""

    ncv: int
    cv_names: List[str]
    cv_types: List[str]
    iwrap: List[int]
    extended_prefix: Optional[str] = None
    extended_args: List[str] = field(default_factory=list)
    kappa_cv: List[float] = field(default_factory=list)
    extended_temp: Optional[float] = None        # TEMP on the EXTENDED_LAGRANGIAN line
    metad: List[MetadInfo] = field(default_factory=list)
    icv_metad: List[int] = field(default_factory=list)
    restraints: List[RestraintInfo] = field(default_factory=list)
    icv_restraint: List[int] = field(default_factory=list)
    w_cv: Optional[int] = None
    colvar_print_args: List[str] = field(default_factory=list)

    @property
    def metad_cv_indices(self) -> List[int]:
        """Sorted list of all CV indices that carry a metadynamics bias."""
        indices = sorted({i for m in self.metad for i in m.cv_indices})
        return indices

    @property
    def has_metad(self) -> bool:
        """True if any metadynamics bias is present in the input."""
        return len(self.metad) > 0


def _plumed_strip_comment(line: str) -> str:
    """Remove anything from the first ``#`` to the end of the line."""
    hash_pos = line.find("#")
    if hash_pos != -1:
        line = line[:hash_pos]
    return line.strip()


def _plumed_logical_lines(text: str) -> List[str]:
    """Return non-empty, comment-stripped lines from raw file text."""
    lines = []
    for raw in text.splitlines():
        cleaned = _plumed_strip_comment(raw)
        if cleaned:
            lines.append(cleaned)
    return lines


def _plumed_parse_keywords(tokens: List[str]) -> Dict[str, str]:
    """Parse ``KEY=value`` tokens into a dict with upper-cased keys."""
    kwargs: Dict[str, str] = {}
    for tok in tokens:
        if "=" in tok:
            key, _, value = tok.partition("=")
            kwargs[key.strip().upper()] = value.strip()
    return kwargs


def _plumed_split_list(value: str) -> List[str]:
    """Split a comma-separated PLUMED value into a list of stripped strings."""
    return [v.strip() for v in value.split(",") if v.strip()]


def _plumed_arg_to_cv_name(arg: str, prefix: Optional[str]) -> str:
    """Map an ARG entry to its base CV name."""
    name = arg.strip()
    if "." in name:
        _, _, remainder = name.partition(".")
        for suffix in ("_vfict", "_fict"):
            if remainder.endswith(suffix):
                return remainder[: -len(suffix)]
        return remainder
    return name


_PLUMED_ACTION_KEYWORDS = {
    "EXTENDED_LAGRANGIAN",
    "METAD",
    "METADYNAMICS",
    "RESTRAINT",
    "PRINT",
}

_PLUMED_LABELLED_RE = re.compile(r"^(?P<label>[^:\s]+)\s*:\s*(?P<rest>.+)$")


def _plumed_validate_colvar_order(
    print_args: List[str], cv_names: List[str], prefix: Optional[str]
) -> None:
    """Ensure the first ``ncv`` printed args are the fictitious vars in CV order."""
    ncv = len(cv_names)
    if len(print_args) < ncv:
        raise ValueError(
            f"PRINT FILE=COLVAR has only {len(print_args)} ARG entries; "
            f"expected at least {ncv} fictitious variables"
        )
    if prefix is None:
        raise ValueError(
            "Cannot validate COLVAR print order: no EXTENDED_LAGRANGIAN prefix "
            "was found before the PRINT line"
        )
    expected = [f"{prefix}.{name}_fict" for name in cv_names]
    got = print_args[:ncv]
    if got != expected:
        raise ValueError(
            "PRINT FILE=COLVAR fictitious variables are out of order: "
            f"got {got}, expected {expected}"
        )


def parse_plumed_dat(path: str) -> PlumedInput:
    """Parse a ``plumed.dat`` file and return a :class:`PlumedInput`."""
    with open(path, "r") as handle:
        lines = _plumed_logical_lines(handle.read())

    cv_names: List[str] = []
    cv_types: List[str] = []
    cv_index: Dict[str, int] = {}

    extended_prefix: Optional[str] = None
    extended_args: List[str] = []
    kappa_cv: List[float] = []
    extended_temp: Optional[float] = None
    metad_list: List[MetadInfo] = []
    restraints: List[RestraintInfo] = []
    w_cv: Optional[int] = None
    colvar_print_args: List[str] = []

    pending: List[tuple] = []
    for line in lines:
        match = _PLUMED_LABELLED_RE.match(line)
        label: Optional[str] = None
        body = line
        if match:
            label = match.group("label")
            body = match.group("rest")

        tokens = body.split()
        if not tokens:
            continue
        action = tokens[0].upper()

        if action in _PLUMED_ACTION_KEYWORDS or (label is None and action == "PRINT"):
            pending.append((label, action, tokens[1:]))
            continue

        if label is None:
            continue
        cv_type = action
        if label in cv_index:
            raise ValueError(f"Duplicate CV name '{label}' in {path}")
        cv_index[label] = len(cv_names)
        cv_names.append(label)
        cv_types.append(cv_type)

    ncv = len(cv_names)
    if ncv == 0:
        raise ValueError(f"No collective variables found in {path}")

    iwrap = [1 if t.upper() == "TORSION" else 0 for t in cv_types]

    def resolve_cv(arg: str) -> int:
        base = _plumed_arg_to_cv_name(arg, extended_prefix)
        if base not in cv_index:
            raise ValueError(
                f"ARG '{arg}' refers to unknown CV '{base}' (known: {cv_names})"
            )
        return cv_index[base]

    for label, action, tokens in pending:
        kwargs = _plumed_parse_keywords(tokens)

        if action == "EXTENDED_LAGRANGIAN":
            extended_prefix = label
            if "ARG" not in kwargs:
                raise ValueError("EXTENDED_LAGRANGIAN line is missing ARG=")
            extended_args = _plumed_split_list(kwargs["ARG"])
            if len(extended_args) != ncv:
                raise ValueError(
                    f"EXTENDED_LAGRANGIAN ARG has {len(extended_args)} entries "
                    f"but there are {ncv} CVs"
                )
            for got, expected in zip(extended_args, cv_names):
                if got != expected:
                    raise ValueError(
                        "EXTENDED_LAGRANGIAN ARG order does not match CV order: "
                        f"got {extended_args}, expected {cv_names}"
                    )
            if "KAPPA" not in kwargs:
                raise ValueError("EXTENDED_LAGRANGIAN line is missing KAPPA=")
            kappa_cv = [float(v) for v in _plumed_split_list(kwargs["KAPPA"])]
            if len(kappa_cv) != ncv:
                raise ValueError(
                    f"EXTENDED_LAGRANGIAN KAPPA has {len(kappa_cv)} entries "
                    f"but there are {ncv} CVs"
                )
            extended_temp = (
                float(kwargs["TEMP"]) if "TEMP" in kwargs else None
            )

        elif action in ("METAD", "METADYNAMICS"):
            if "ARG" not in kwargs:
                raise ValueError("METAD line is missing ARG=")
            args = _plumed_split_list(kwargs["ARG"])
            indices = [resolve_cv(a) for a in args]
            names = [cv_names[i] for i in indices]
            if "PACE" not in kwargs:
                raise ValueError("METAD line is missing PACE=")
            w_hill = int(kwargs["PACE"])
            if "TEMP" not in kwargs:
                print(
                    f"!!WARNING: TEMP keyword is missing along with METAD "
                    f"keywords in {path}",
                    file=sys.stderr,
                )
                temp = None
            else:
                temp = float(kwargs["TEMP"])
            biasfactor = (
                float(kwargs["BIASFACTOR"]) if "BIASFACTOR" in kwargs else None
            )
            height = float(kwargs["HEIGHT"]) if "HEIGHT" in kwargs else None
            sigma = (
                [float(v) for v in _plumed_split_list(kwargs["SIGMA"])]
                if "SIGMA" in kwargs
                else None
            )
            metad_list.append(
                MetadInfo(
                    label=label or "metad",
                    cv_indices=indices,
                    cv_names=names,
                    w_hill=w_hill,
                    biasfactor=biasfactor,
                    temp=temp,
                    height=height,
                    sigma=sigma,
                )
            )

        elif action == "RESTRAINT":
            if "ARG" not in kwargs:
                raise ValueError("RESTRAINT line is missing ARG=")
            args = _plumed_split_list(kwargs["ARG"])
            if "KAPPA" not in kwargs:
                raise ValueError("RESTRAINT line is missing KAPPA=")
            if "AT" not in kwargs:
                raise ValueError("RESTRAINT line is missing AT=")
            kappa_vals = [float(v) for v in _plumed_split_list(kwargs["KAPPA"])]
            at_vals = [float(v) for v in _plumed_split_list(kwargs["AT"])]
            for j, arg in enumerate(args):
                idx = resolve_cv(arg)
                kappa = kappa_vals[j] if j < len(kappa_vals) else kappa_vals[-1]
                at = at_vals[j] if j < len(at_vals) else at_vals[-1]
                restraints.append(
                    RestraintInfo(
                        label=label or "restraint",
                        cv_index=idx,
                        cv_name=cv_names[idx],
                        kappa_restraint=kappa,
                        restraint_at=at,
                    )
                )

        elif action == "PRINT":
            if kwargs.get("FILE", "").upper() == "COLVAR":
                if "STRIDE" not in kwargs:
                    raise ValueError("PRINT FILE=COLVAR line is missing STRIDE=")
                w_cv = int(kwargs["STRIDE"])
                colvar_print_args = _plumed_split_list(kwargs.get("ARG", ""))
                _plumed_validate_colvar_order(
                    colvar_print_args, cv_names, extended_prefix
                )

    icv_metad = [0] * ncv
    for m in metad_list:
        for idx in m.cv_indices:
            icv_metad[idx] = 1

    icv_restraint = [0] * ncv
    for r in restraints:
        icv_restraint[r.cv_index] = 1

    return PlumedInput(
        ncv=ncv,
        cv_names=cv_names,
        cv_types=cv_types,
        iwrap=iwrap,
        extended_prefix=extended_prefix,
        extended_args=extended_args,
        kappa_cv=kappa_cv,
        extended_temp=extended_temp,
        metad=metad_list,
        icv_metad=icv_metad,
        restraints=restraints,
        icv_restraint=icv_restraint,
        w_cv=w_cv,
        colvar_print_args=colvar_print_args,
    )


# =========================================================================== #
#  Parsing: tass.inp + per-window plumed.dat
# =========================================================================== #
def _restraint_cv_index(plumed) -> int:
    """0-based index of the single CV carrying the umbrella RESTRAINT."""
    idx = [i for i, r in enumerate(plumed.icv_restraint) if r == 1]
    if len(idx) != 1:
        sys.exit(
            f"!!ERROR: TASS expects exactly one RESTRAINT CV per window, "
            f"found {len(idx)} (icv_restraint={plumed.icv_restraint})"
        )
    return idx[0]


def _restraint_params(plumed, cv1: int):
    """(kappa_restraint, restraint_at) for the restraint acting on CV `cv1`."""
    for r in plumed.restraints:
        if r.cv_index == cv1:
            return r.kappa_restraint, r.restraint_at
    sys.exit(f"!!ERROR: no RESTRAINT found acting on CV index {cv1}")


def _metad_params(plumed):
    """(w_hill, biasfactor) from the METAD action(s), requiring agreement.

    Returns (None, None) when the window has no METAD action.  The temperature
    is taken from the EXTENDED_LAGRANGIAN line, not from here.
    """
    if not plumed.metad:
        return None, None
    w_hills = {m.w_hill for m in plumed.metad}
    bfs = {m.biasfactor for m in plumed.metad}
    if len(w_hills) != 1:
        sys.exit(f"!!ERROR: inconsistent PACE across METAD lines: {w_hills}")
    if len(bfs) != 1:
        sys.exit(f"!!ERROR: inconsistent BIASFACTOR across METAD lines: {bfs}")
    bf = bfs.pop()
    if bf is None:
        sys.exit("!!ERROR: BIASFACTOR is required on the METAD line for reweighting")
    return w_hills.pop(), bf


def parse_tass(text: str, root: Path) -> dict:
    """Read tass.inp + each window's plumed.dat into one config dict."""
    rdr = InputReader(text)
    nwin = rdr.one_int("nwin")
    if nwin < 1:
        sys.exit("!!ERROR: number of windows must be >= 1")

    windows: list[dict] = []

    for iw in range(nwin):
        folder = rdr.tokens(f"window folder #{iw + 1}")[0]
        plumed_path = root / folder / "plumed.dat"
        if not plumed_path.is_file():
            sys.exit(f"!!ERROR: plumed.dat not found for window {iw + 1}: {plumed_path}")
        plumed = parse_plumed_dat(str(plumed_path))

        # Per-window CV layout (may differ across windows).
        ncv = plumed.ncv
        iwrap = list(plumed.iwrap)
        icv_restraint = list(plumed.icv_restraint)
        cv1 = _restraint_cv_index(plumed)

        midx = [i for i, m in enumerate(plumed.icv_metad) if m == 1]
        has_metad = len(midx) > 0

        kappa_restraint, restraint_at = _restraint_params(plumed, cv1)
        w_hill, biasfactor = _metad_params(plumed)
        if plumed.extended_temp is None:
            sys.exit(f"!!ERROR: TEMP not found on the EXTENDED_LAGRANGIAN line "
                     f"in window {iw + 1} ({plumed_path})")
        temp = plumed.extended_temp
        if plumed.w_cv is None:
            sys.exit(f"!!ERROR: no PRINT ... FILE=COLVAR STRIDE found "
                     f"(w_cv) in window {iw + 1}")
        w_cv = plumed.w_cv

        # --- metad grid mode line (only for windows that HAVE metad) ----- #
        metad_grids: dict | None = None
        auto_metad = False
        if has_metad:
            mode_tok = rdr.tokens(f"metad grid mode (window {iw + 1})")
            mode = mode_tok[0].upper()
            ncv_mtd = len(midx)
            if mode == "GRIDS_METAD":
                metad_grids = {}
                for _ in range(ncv_mtd):
                    tk = rdr.tokens(f"metad grid (window {iw + 1})")
                    if len(tk) < 4:
                        sys.exit(f"!!ERROR: GRIDS_METAD line needs "
                                 f"'icv_mtd gmin gmax dgrid' (window {iw + 1})")
                    icv1b = int(tk[0])
                    gmin, gmax, dg = (_to_float(tk[1]), _to_float(tk[2]),
                                      _to_float(tk[3]))
                    metad_grids[icv1b - 1] = (gmin, gmax, dg)
                missing = [i for i in midx if i not in metad_grids]
                if missing:
                    sys.exit(f"!!ERROR: GRIDS_METAD missing grids for metad CV "
                             f"indices (1-based) {[i + 1 for i in missing]} "
                             f"in window {iw + 1}")
            elif mode == "AUTO_GRIDS_METAD":
                auto_metad = True
            else:
                sys.exit(f"!!ERROR: expected GRIDS_METAD or AUTO_GRIDS_METAD "
                         f"in window {iw + 1}, got '{mode}'")

        t_min, t_max = rdr.ints(2, f"t_min t_max (window {iw + 1})")

        windows.append(dict(
            folder=folder, ncv=ncv, cv_names=list(plumed.cv_names),
            iwrap=iwrap, icv_restraint=icv_restraint,
            cv1=cv1, t_min=t_min, t_max=t_max, w_cv=w_cv, w_hill=w_hill,
            biasfactor=biasfactor, temp=temp, kappa_restraint=kappa_restraint,
            restraint_at=restraint_at, has_metad=has_metad, midx=midx,
            auto_metad=auto_metad, metad_grids=metad_grids,
            plumed_path=str(plumed_path),
        ))

    # --- optional PRINT2D section (after all windows) -------------------- #
    print_2d = False
    auto2 = False
    icv2 = None
    grid2 = None
    marker = rdr.raw_line()
    if marker is not None and "PRINT2D" in marker.upper():
        print_2d = True
        auto2 = "AUTO_GRID" in marker.upper()
        tk = rdr.tokens("PRINT2D grid line (icv2 [gmin gmax dgrid])")
        icv2 = int(tk[0])                          # 1-based CV index (per window)
        for iw, win in enumerate(windows, start=1):
            if not (1 <= icv2 <= win["ncv"]):
                sys.exit(f"!!ERROR: icv2={icv2} out of range 1..ncv "
                         f"({win['ncv']}) for window {iw}")
            if icv2 - 1 == win["cv1"]:
                sys.exit(f"!!ERROR: icv2 cannot be the restraint CV (cv1) "
                         f"in window {iw}")
        if not auto2:
            if len(tk) < 4:
                sys.exit("!!ERROR: PRINT2D without AUTO_GRID needs "
                         "'icv2 gmin gmax dgrid'")
            grid2 = (_to_float(tk[1]), _to_float(tk[2]), _to_float(tk[3]))

    return dict(
        nwin=nwin,
        windows=windows, print_2d=print_2d, auto2=auto2, icv2=icv2, grid2=grid2,
    )


def _report_tass_input(inp_path: Path, root: Path, cfg: dict) -> None:
    """Print a summary of ``tass.inp``, each ``plumed.dat``, and global options."""
    print("=" * 70)
    print("Input summary")
    print("=" * 70)
    print(f"  control file : {inp_path.resolve()}")
    print(f"  root         : {root.resolve()}")
    print(f"  nwin         : {cfg['nwin']}")

    for iw, win in enumerate(cfg["windows"], start=1):
        print("-" * 70)
        print(f"Window {iw}/{cfg['nwin']}: {win['folder']}")
        print(f"  plumed.dat   : {win['plumed_path']}")
        cv_list = ", ".join(
            f"{n}({'TORSION' if win['iwrap'][j] else 'non-periodic'})"
            for j, n in enumerate(win["cv_names"])
        )
        cv1_name = win["cv_names"][win["cv1"]]
        print(f"    ncv        : {win['ncv']}")
        print(f"    CVs        : {cv_list}")
        print(f"    cv1        : {win['cv1'] + 1} ({cv1_name})  "
              f"icv_restraint={win['icv_restraint']}")
        print(f"    RESTRAINT  : kappa={win['kappa_restraint']:.4g}  "
              f"AT={win['restraint_at']:.6g}")
        print(f"    EXTENDED   : TEMP={win['temp']:.4g}  "
              f"w_cv(STRIDE)={win['w_cv']}")
        if win["has_metad"]:
            print(f"    METAD      : CVs (1-based)={[i + 1 for i in win['midx']]}  "
                  f"PACE={win['w_hill']}  BIASFACTOR={win['biasfactor']:.4g}")
            if win["auto_metad"]:
                print("    metad grid : AUTO_GRIDS_METAD (from COLVAR at run time)")
            else:
                for icv, (gmin, gmax, dg) in sorted(win["metad_grids"].items()):
                    nb = nbins_from_grid(gmin, gmax, dg)
                    print(f"    metad grid : CV {icv + 1}  "
                          f"gmin={gmin:.6g}  gmax={gmax:.6g}  "
                          f"dgrid={dg:.6g}  nbin={nb}")
        else:
            print("    METAD      : (none)  reweighting with vbias=0, c(t)=0")
        tmax_lbl = ("all COLVAR frames" if win["t_max"] < 0
                    else str(win["t_max"]))
        print(f"  tass.inp     : t_min={win['t_min']}  t_max={win['t_max']} "
              f"({tmax_lbl})")

    if cfg["print_2d"]:
        print("-" * 70)
        print("PRINT2D (global)")
        print(f"  enabled      : yes")
        print(f"  icv2         : {cfg['icv2']} (1-based, per-window CV index)")
        if cfg["auto2"]:
            print("  grid         : AUTO_GRID (from pooled COLVAR at run time)")
        else:
            g2 = cfg["grid2"]
            print(f"  grid         : gmin={g2[0]:.6g}  gmax={g2[1]:.6g}  "
                  f"dgrid={g2[2]:.6g}  nbin={nbins_from_grid(*g2)}")
    else:
        print("-" * 70)
        print("PRINT2D        : (not requested)")
    print("=" * 70)


# =========================================================================== #
#  Auto grids
# =========================================================================== #
def _auto_grid(values: np.ndarray, label: str = "CV"):
    """(gmin, gmax, dgrid) with AUTO_NBINS bins spanning the data range.

    Uses linear ``min`` / ``max`` of the sampled values.  If ``gmax <= gmin``
    the program stops with an error.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        sys.exit(f"!!ERROR: cannot build auto grid for {label}: no data")

    gmin = float(np.min(v))
    gmax = float(np.max(v))
    if gmax <= gmin:
        sys.exit(
            f"!!ERROR: degenerate auto grid for {label}: "
            f"gmax ({gmax}) <= gmin ({gmin}); cannot proceed"
        )
    dgrid = (gmax - gmin) / AUTO_NBINS
    return gmin, gmax, dgrid


def resolve_metad_grids(folder: Path, cfg: dict, win: dict) -> dict:
    """Return {cv_index: (gmin, gmax, dgrid)} for the metad CVs of this window."""
    if not win["auto_metad"]:
        return win["metad_grids"]
    midx = win["midx"]
    cols = tuple(i + 1 for i in midx)              # COLVAR cols (time = col 0)
    cv = np.loadtxt(folder / "COLVAR", comments="#", usecols=cols, ndmin=2)
    grids = {}
    for k, i in enumerate(midx):
        label = f"metad CV {i + 1} (window {win['folder']})"
        grids[i] = _auto_grid(cv[:, k], label=label)
    return grids


# =========================================================================== #
#  Step 1: c(t) factor  ->  ct.dat   (serial)
# =========================================================================== #
def run_ct_factor(folder: Path, cfg: dict, win: dict,
                  md_steps: int, mtd_steps: int) -> None:
    t0 = time.time()
    kt_energy = KB * win["temp"]
    gamma = (win["biasfactor"] - 1.0) / win["biasfactor"]

    midx = win["midx"]
    ncv_mtd = len(midx)
    grids = resolve_metad_grids(folder, cfg, win)
    for i in midx:
        gmin, gmax, dg = grids[i]
        print(f"      ct grid CV {i + 1}: gmin={gmin:.6g}  gmax={gmax:.6g}  "
              f"dgrid={dg:.6g}  nbin={nbins_from_grid(gmin, gmax, dg)}")
    gmin = np.array([grids[i][0] for i in midx], dtype=np.float64)
    gmax = np.array([grids[i][1] for i in midx], dtype=np.float64)
    nbin = np.array([nbins_from_grid(*grids[i]) for i in midx], dtype=np.int64)
    iwrap = np.array([win["iwrap"][i] for i in midx], dtype=np.int64)

    hills_path = folder / "HILLS"
    print(f"      reading HILLS : {hills_path}  ({mtd_steps} hills, "
          f"{ncv_mtd} metad CVs)")
    hill, width, ht = ct_read_hills(hills_path, mtd_steps, ncv_mtd)
    coords = build_grid_coords(gmin, gmax, nbin)
    print(f"      ct grid points: {coords.shape[0]}")
    ct = compute_ct(hill, width, ht, coords, kt_energy, gamma, iwrap)

    out = folder / "ct.dat"
    np.savetxt(out,
               np.column_stack([np.arange(1, mtd_steps + 1, dtype=np.int64), ct]),
               fmt="%10d %16.8f")
    elapsed = time.time() - t0
    print(f"    ct.dat       : COMPUTED -> {out}  "
          f"(mtd_steps={mtd_steps}, {elapsed:.2f}s)")


# =========================================================================== #
#  Step 2: bias potential  ->  vbias.dat   (MPI worker / in-process)
# =========================================================================== #
def compute_vbias_window(folder: Path, cfg: dict, win: dict, md_steps: int,
                         threads: int, backend: str, comm=None) -> None:
    """Compute vbias.dat for one window; serial (comm=None) or MPI (comm)."""
    rank, nranks = (comm.Get_rank(), comm.Get_size()) if comm is not None else (0, 1)
    parent = (rank == 0)

    midx = np.array(win["midx"], dtype=np.int64)
    ncv_mtd = midx.size
    gamma = (win["biasfactor"] - 1.0) / win["biasfactor"]
    iwrap = np.ascontiguousarray(
        np.array([win["iwrap"][i] for i in midx], dtype=np.int64))
    w_cv, w_hill = win["w_cv"], win["w_hill"]
    t_max = md_steps                              # vbias for every COLVAR frame

    if parent:
        cv = vbias_load_colvar(folder / "COLVAR", midx)
        centers, widths, heights = vbias_load_hills(folder / "HILLS", ncv_mtd)
    else:
        cv = centers = widths = heights = None
    if comm is not None:
        cv = comm.bcast(cv, root=0)
        centers = comm.bcast(centers, root=0)
        widths = comm.bcast(widths, root=0)
        heights = comm.bcast(heights, root=0)

    inv_sig2 = np.ascontiguousarray(1.0 / (widths * widths))
    ht_gamma = np.ascontiguousarray(heights * gamma)

    all_steps = np.arange(1, t_max + 1, dtype=np.int64)
    my_steps = all_steps[rank::nranks]

    kernel = _build_numba_kernel() if backend == "numba" else None
    if kernel is not None:
        if threads > 0:
            try:
                import numba
                numba.set_num_threads(threads)
            except Exception:
                pass
        my_vbias = kernel(my_steps, cv, centers, inv_sig2, ht_gamma, iwrap,
                          w_cv, w_hill)
    else:
        my_vbias = vbias_steps_numpy(my_steps, cv, centers, inv_sig2, ht_gamma,
                                     iwrap, w_cv, w_hill)

    if comm is not None and nranks > 1:
        gathered_steps = comm.gather(my_steps, root=0)
        gathered_vbias = comm.gather(my_vbias, root=0)
    else:
        gathered_steps, gathered_vbias = [my_steps], [my_vbias]

    if parent:
        vbias = np.empty(t_max, dtype=np.float64)
        for s, v in zip(gathered_steps, gathered_vbias):
            vbias[s - 1] = v
        ref = vbias[0]
        with (folder / "vbias.dat").open("w") as fh:
            for i_md in range(1, t_max + 1):
                fh.write(f"{i_md:10d}{vbias[i_md - 1] - ref:15.6f}\n")


def run_vbias(folder: Path, cfg: dict, win: dict, inp_path: Path, root: Path,
              iw_index: int, md_steps: int, nranks: int, threads: int,
              backend: str, use_mpi: bool, env: dict) -> None:
    t0 = time.time()
    if use_mpi and nranks > 1:
        cmd = ["mpirun",
               "--mca", "orte_tmpdir_base", env.get("TMPDIR", "/tmp"),
               "--mca", "pmix_server_tmpdir", env.get("TMPDIR", "/tmp"),
               "-np", str(nranks),
               sys.executable, str(SCRIPT_PATH), str(inp_path),
               "--vbias-worker", "--window-index", str(iw_index),
               "--root", str(root), "--threads", str(threads),
               "--backend", backend]
        print(f"    vbias.dat    : COMPUTING (mpirun -np {nranks} x {threads} "
              f"threads, backend={backend}) ...")
        subprocess.run(cmd, env=env, check=True)
    else:
        print(f"    vbias.dat    : COMPUTING (1 rank x {threads} threads, "
              f"backend={backend}) ...")
        compute_vbias_window(folder, cfg, win, md_steps, threads, backend, comm=None)
    elapsed = time.time() - t0
    print(f"    vbias.dat    : COMPUTED -> {folder / 'vbias.dat'}  "
          f"(md_steps={md_steps}, {elapsed:.2f}s)")


def vbias_worker(args) -> int:
    """Hidden mode: one MPI rank computing vbias.dat for a single window."""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    except Exception:
        comm = None

    root = Path(args.root)
    cfg = parse_tass(Path(args.input).read_text(), root)
    win = cfg["windows"][args.window_index]
    folder = root / win["folder"]
    md_steps = count_lines(folder / "COLVAR")
    compute_vbias_window(folder, cfg, win, md_steps, args.threads, args.backend,
                         comm=comm)
    return 0


# =========================================================================== #
#  Step 3: reweighted mean force at each window's restraint centre
# =========================================================================== #
def _pbc(diff: np.ndarray, periodic: bool) -> np.ndarray:
    """Minimum-image difference with period 2*pi for periodic (TORSION) CVs."""
    if periodic:
        return diff - TWOPI * np.rint(diff / TWOPI)
    return diff


def _reweight_weights(folder: Path, win: dict, i_md: np.ndarray,
                      md_steps: int) -> np.ndarray:
    """Reweighting weight exp((V_bias - c(t))/kT) for the MD frames `i_md`.

    Windows without a metadynamics bias use V_bias = c(t) = 0, i.e. weight = 1
    for every frame; ct.dat / vbias.dat are not read in that case.
    """
    if not win["has_metad"]:
        return np.ones(i_md.size, dtype=np.float64)
    kt = KB * win["temp"]
    vbias = np.loadtxt(folder / "vbias.dat", ndmin=2)[:, 1]
    ct = np.loadtxt(folder / "ct.dat", ndmin=2)[:, 1]
    mtd_steps = ct.shape[0]
    i_mtd = np.clip((i_md * win["w_cv"]) // win["w_hill"], 1, mtd_steps)
    return np.exp((vbias[i_md - 1] - ct[i_mtd - 1]) / kt)


def compute_mean_force(folder: Path, win: dict, md_steps: int) -> float:
    """Reweighted < -k*(s1(t) - s1_at) > at this window's restraint centre."""
    cv1 = win["cv1"]
    cv1_col = cv1 + 1                              # COLVAR col (time = col 0)
    periodic = bool(win["iwrap"][cv1])

    t_min = win["t_min"]
    t_max = md_steps if win["t_max"] < 0 else win["t_max"]
    if t_max > md_steps:
        sys.exit(f"!!ERROR: t_max ({t_max}) > md_steps ({md_steps}) in {folder}")

    colvar = np.loadtxt(folder / "COLVAR", comments="#", ndmin=2)[:md_steps]
    s1 = colvar[:, cv1_col]

    i_md = np.arange(t_min, t_max + 1)
    weight = _reweight_weights(folder, win, i_md, md_steps)
    diff = _pbc(s1[i_md - 1] - win["restraint_at"], periodic)
    force = -win["kappa_restraint"] * diff

    den = float(weight.sum())
    if den <= 0.0:
        sys.exit(f"!!ERROR: zero total weight while averaging mean force in {folder}")
    return float(np.sum(weight * force) / den)


# =========================================================================== #
#  Step 4: F1(s1) by trapezoidal integration of the mean force
# =========================================================================== #
def integrate_mean_force(s1: np.ndarray, dfds: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral of dF/ds over (possibly non-uniform) s1."""
    fes1 = np.zeros_like(s1)
    if s1.size > 1:
        ds = np.diff(s1)
        incr = 0.5 * ds * (dfds[:-1] + dfds[1:])
        fes1[1:] = np.cumsum(incr)
    return fes1


# =========================================================================== #
#  Step 5: 2D free energy  F(cv1, cv2)
# =========================================================================== #
def _bin_index(values: np.ndarray, gmin: float, dgrid: float,
               periodic: bool) -> np.ndarray:
    """1-based bin index; periodic CVs are folded into [gmin, gmin+2*pi)."""
    d = values - gmin
    if periodic:
        d = d - TWOPI * np.floor(d / TWOPI)
    return (_nint(d / dgrid) + 1).astype(int)


def compute_fes_2d(cfg: dict, root: Path, order: list[int], s1_grid: np.ndarray,
                   fes1: np.ndarray, grid2: tuple) -> None:
    """Stitch a 2D surface F(cv1, cv2): one cv1 row per window (sorted)."""
    icv2 = cfg["icv2"]                             # 1-based CV index (per window)
    cv2_col = icv2                                 # COLVAR col == 1-based CV index
    gmin2, gmax2, dg2 = grid2
    ngrid2 = nbins_from_grid(gmin2, gmax2, dg2)
    ngrid1 = len(order)

    prob = np.zeros((ngrid1, ngrid2))
    norm = np.zeros(ngrid1)

    for row, iw in enumerate(order):
        win = cfg["windows"][iw]
        folder = root / win["folder"]
        periodic2 = bool(win["iwrap"][icv2 - 1])
        colvar = load_table(folder / "COLVAR", f"COLVAR (window {iw + 1})",
                            min_cols=win["ncv"] + 1)

        md_steps = colvar.shape[0]
        if win["has_metad"]:
            vbias = load_table(folder / "vbias.dat",
                               f"vbias.dat (window {iw + 1})", min_cols=2)[:, 1]
            if vbias.shape[0] != md_steps:
                sys.exit(f"!!ERROR: md_steps mismatch (window {iw + 1}): "
                         f"vbias.dat has {vbias.shape[0]}, COLVAR has {md_steps}")

        t_max = md_steps if win["t_max"] < 0 else win["t_max"]
        i_md = np.arange(1, md_steps + 1)
        cv2 = colvar[:, cv2_col]
        jbin = _bin_index(cv2, gmin2, dg2, periodic2)

        ok_jbin = (jbin >= 1) & (jbin <= ngrid2 - 1)
        ok_time = (i_md >= win["t_min"]) & (i_md <= t_max)
        mask = ok_jbin & ok_time

        if np.any(mask):
            sel = np.where(mask)[0]
            weight = _reweight_weights(folder, win, i_md[sel], md_steps)
            np.add.at(prob[row], jbin[sel] - 1, weight)
            norm[row] += float(weight.sum())

    print(" (Info) 2D probability is computed.")

    with np.errstate(divide="ignore", invalid="ignore"):
        inv_norm = 1.0 / (norm * dg2)
    inv_norm[~np.isfinite(inv_norm)] = 0.0

    s2_grid = gmin2 + np.arange(ngrid2) * dg2
    kt0 = KB * cfg["windows"][order[0]]["temp"]
    with (Path.cwd() / "free_energy_2D.dat").open("w") as fh:
        for ibin in range(ngrid1):
            for jbin in range(ngrid2):
                p_cond = prob[ibin, jbin] * inv_norm[ibin]
                fe = -kt0 * np.log(max(p_cond, MIN_PROB)) + fes1[ibin]
                fh.write(f"{s1_grid[ibin]:16.8E}{s2_grid[jbin]:16.8E}"
                         f"{fe:16.8E}{p_cond:16.8E}\n")
            fh.write("\n")
    print(" (Info) 2D free energy + unbiased distribution -> free_energy_2D.dat "
          "(kJ/mol)")


# =========================================================================== #
#  Main driver
# =========================================================================== #
def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", default="tass.inp",
                        help="control input file (default: tass.inp)")
    parser.add_argument("--root", default=".",
                        help="base directory the window folders are under (default: cwd)")
    parser.add_argument("--nranks", type=int, default=1,
                        help="MPI ranks for the vbias step (default: 1)")
    parser.add_argument("--threads", type=int, default=1,
                        help="threads per rank for vbias (default: 1)")
    parser.add_argument("--backend", choices=["numpy", "numba"], default="numpy",
                        help="vbias shared-memory kernel (default: numpy)")
    parser.add_argument("--force", action="store_true",
                        help="recompute ct.dat / vbias.dat even if they already exist")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="verbose output")
    parser.add_argument("--vbias-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--window-index", type=int, default=0, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.vbias_worker:
        return vbias_worker(args)

    inp_path = Path(args.input)
    if not inp_path.is_file():
        sys.exit(f"!!ERROR: control file not found: {inp_path}")
    root = Path(args.root)

    cfg = parse_tass(inp_path.read_text(), root)
    _report_tass_input(inp_path, root, cfg)

    print("=" * 70)
    print("Run configuration")
    print("=" * 70)
    print(f"  MPI vbias    : {'yes' if args.nranks > 1 else 'no'}  "
          f"(nranks={args.nranks}, threads={args.threads}, "
          f"backend={args.backend})")
    print(f"  force recompute: {args.force}")

    have_mpi4py = importlib.util.find_spec("mpi4py") is not None
    have_mpirun = shutil.which("mpirun") is not None
    use_mpi = args.nranks > 1 and have_mpi4py and have_mpirun
    if args.nranks > 1 and not use_mpi:
        print(f"  [warn] --nranks {args.nranks} requested but "
              f"{'mpi4py' if not have_mpi4py else 'mpirun'} unavailable -> "
              f"running vbias as a single threaded process.")

    env = os.environ.copy()
    for var in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS",
                "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[var] = str(args.threads)
    env.setdefault("TMPDIR", env.get("MPI_TMPDIR", "/tmp"))

    t0 = time.time()
    mean_forces = np.empty(cfg["nwin"], dtype=np.float64)
    restraint_centres = np.empty(cfg["nwin"], dtype=np.float64)

    for iw, win in enumerate(cfg["windows"], start=1):
        folder = root / win["folder"]
        print("-" * 70)
        metad_label = (f"metad CVs {[i + 1 for i in win['midx']]}"
                       if win["has_metad"] else "NO metad (vbias=0, c(t)=0)")
        print(f"[window {iw}/{cfg['nwin']}] {folder}")
        print(f"    {metad_label}")
        if not folder.is_dir():
            sys.exit(f"!!ERROR: window folder not found: {folder}")

        colvar_path = folder / "COLVAR"
        if not colvar_path.is_file():
            sys.exit(f"!!ERROR: missing COLVAR in {folder}")
        md_steps = count_lines(colvar_path)
        print(f"    COLVAR       : {colvar_path}  ({md_steps} frames)")

        if win["has_metad"]:
            hills_path = folder / "HILLS"
            if not hills_path.is_file():
                sys.exit(f"!!ERROR: missing HILLS in {folder}")
            mtd_steps = count_lines(hills_path)
            print(f"    HILLS        : {hills_path}  ({mtd_steps} hills)")

            if (folder / "ct.dat").is_file() and not args.force:
                print(f"    ct.dat       : REUSED (existing file, use --force "
                      f"to recompute)")
            else:
                print("    ct.dat       : computing ...")
                run_ct_factor(folder, cfg, win, md_steps, mtd_steps)

            if (folder / "vbias.dat").is_file() and not args.force:
                print(f"    vbias.dat    : REUSED (existing file, use --force "
                      f"to recompute)")
            else:
                run_vbias(folder, cfg, win, inp_path, root, iw - 1, md_steps,
                          args.nranks, args.threads, args.backend, use_mpi, env)
        else:
            print("    no METAD     : skipping ct.dat / vbias.dat "
                  "(reweighting with vbias=0, c(t)=0)")

        t0_mf = time.time()
        mf = compute_mean_force(folder, win, md_steps)
        elapsed_mf = time.time() - t0_mf
        mean_forces[iw - 1] = mf
        restraint_centres[iw - 1] = win["restraint_at"]
        print(f"    mean force   : dF1/ds1 = {mf:.6f}  at s1 = "
              f"{win['restraint_at']:.4f}  ({elapsed_mf:.2f}s)")

    print("-" * 70)
    print(f"All per-window calculations done in {time.time() - t0:.1f}s")

    # --- 1D: F1(s1) by trapezoidal integration of the mean force --------- #
    order = list(np.argsort(restraint_centres))   # ascending restraint centre
    s1_grid = restraint_centres[order]
    dfds = mean_forces[order]
    if np.any(np.diff(s1_grid) <= 0.0):
        sys.exit("!!ERROR: two windows share the same restraint centre; "
                 "cannot integrate the mean force")
    fes1 = integrate_mean_force(s1_grid, dfds)

    with (Path.cwd() / "mean_force.dat").open("w") as fh:
        fh.write("#       s1            dF1/ds1            F1\n")
        for s, g, f in zip(s1_grid, dfds, fes1):
            fh.write(f"{s:16.8f}{g:16.8f}{f:16.8f}\n")
    with (Path.cwd() / "fes_1d.dat").open("w") as fh:
        for s, f in zip(s1_grid, fes1):
            fh.write(f" {s:.10g}   {f:.10g}\n")
    print(" (Info) Mean force -> mean_force.dat ; F1(s1) -> fes_1d.dat")

    # --- 2D: optional F(cv1, cv2) --------------------------------------- #
    if cfg["print_2d"]:
        print("=" * 70)
        print("Computing 2D free energy surface ...")
        if cfg["auto2"]:
            icv2 = cfg["icv2"]
            pooled = []
            for win in cfg["windows"]:
                folder = root / win["folder"]
                pooled.append(np.loadtxt(folder / "COLVAR", comments="#",
                                         usecols=(icv2,), ndmin=1))
            grid2 = _auto_grid(np.concatenate(pooled),
                               label=f"CV {icv2} (PRINT2D AUTO_GRID)")
            print(f"  AUTO_GRID for CV {icv2}: gmin={grid2[0]:.4f} "
                  f"gmax={grid2[1]:.4f} dgrid={grid2[2]:.4f}")
        else:
            grid2 = cfg["grid2"]
        compute_fes_2d(cfg, root, order, s1_grid, fes1, grid2)
    else:
        print(" Warning! 2D free energy will not be printed (no PRINT2D).")

    print("=" * 70)
    print("Pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
