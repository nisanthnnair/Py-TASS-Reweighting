# `tass_reweight.py` — User Manual

Multi-window **TASS** (Temperature-Accelerated Sliced Sampling) reweighting and
free-energy reconstruction from PLUMED outputs.

`tass_reweight.py` reads the simulation parameters directly from each window's
`plumed.dat`, computes the metadynamics reweighting factors (`c(t)` and
`V_bias`), and stitches the umbrella windows into a 1D free-energy profile and an
optional 2D free-energy surface.

---

## 1. Synopsis

```bash
python tass_reweight.py [tass.inp] [--root DIR]
                        [--nranks N] [--threads T] [--backend numpy|numba]
                        [--force] [-v]
```

Defaults: control file `tass.inp`, `--root .`, serial (`--nranks 1 --threads 1`),
`--backend numpy`.

---

## 2. Requirements

- Python 3.8+
- [NumPy](https://numpy.org/) only — `tass_reweight.py` is a single self-contained
  file (the `plumed.dat` parser and the `c(t)`/`V_bias` kernels are all inside it).
  `plumed_parser.py` is an optional thin re-export for backward compatibility.
- Optional accelerators for the `vbias` step:
  - [`mpi4py`](https://mpi4py.readthedocs.io/) + an MPI runtime (`mpirun`) for `--nranks > 1`
  - [`numba`](https://numba.pydata.org/) for `--backend numba`

```bash
python -m pip install numpy              # core
python -m pip install mpi4py numba       # optional
```

> If `python -c "import numpy"` fails or crashes, run inside a clean virtual
> environment: `python -m venv .venv && source .venv/bin/activate && pip install numpy`.

Conventions: energies in **kJ/mol** (`kB = 8.314472e-3 kJ/mol/K`); periodic CVs
(PLUMED `TORSION`) use period `2*pi`.

---

## 3. Inputs

### 3.1 Directory layout

One folder per umbrella window, each holding the three PLUMED outputs:

```
<root>/
├── tass.inp
├── W1/
│   ├── plumed.dat
│   ├── COLVAR
│   └── HILLS
├── W2/
│   ├── plumed.dat
│   ├── COLVAR
│   └── HILLS
└── ...
```

Window folder names are taken from `tass.inp` and resolved relative to `--root`.

### 3.2 What is read from each `plumed.dat`

The driver does **not** take physical parameters from the control file; it reads
them per window from `plumed.dat`:

| Quantity | Source line in `plumed.dat` |
|----------|-----------------------------|
| `ncv`, CV names, `iwrap` (TORSION→1) | the `name: TYPE ...` CV declarations |
| `TEMP` (the temperature TASS uses) | `EXTENDED_LAGRANGIAN` |
| `icv_metad` (which CV has MetaD), `w_hill` (PACE), `BIASFACTOR` | `METAD` / `METADYNAMICS` (optional) |
| `icv_restraint` (which CV is restrained), `kappa_restraint` (KAPPA), `restraint_at` (AT) | `RESTRAINT` |
| `w_cv` (STRIDE) and the COLVAR column order | `PRINT ... FILE=COLVAR` |

COLVAR column layout implied by the `PRINT` line: column `0` = time, column
`i+1` = fictitious value of CV `i` (`ex.<cv>_fict`).

**Temperature** is read from the `TEMP` keyword on the `EXTENDED_LAGRANGIAN`
line (mandatory). `TEMP` on a `METAD` line is ignored; if a `METAD` line is
present but has no `TEMP`, the parser prints a warning (otherwise harmless).

**The `METAD` action is optional.** A window whose `plumed.dat` has no `METAD`
needs no `HILLS` and produces no `ct.dat` / `vbias.dat`; its reweighting weight
is `1` for every frame (equivalent to `V_bias = 0`, `c(t) = 0`).

**Per-window layout:** `ncv`, `iwrap`, `icv_restraint` and the restraint CV index
`cv1` are read independently from each window's `plumed.dat` and may differ
between windows. Within a window, `cv1` is the 0-based index of the CV carrying
the umbrella `RESTRAINT` (the entry of `icv_restraint` equal to 1). `icv_metad`
may also differ per window.

**`PRINT2D` `icv2`:** the 1-based CV index in `tass.inp` is interpreted within
each window's own CV layout (validated per window: `1 <= icv2 <= ncv` and
`icv2 != cv1`).

### 3.3 The `tass.inp` control file

```text
<nwin>                              # number of windows (first non-blank line)

# --- repeated nwin times ---
<folder>                           # folder with plumed.dat, COLVAR (+ HILLS if metad)
GRIDS_METAD | AUTO_GRIDS_METAD     # metad-grid mode; OMIT for a no-metad window
  <icv_mtd> <gmin> <gmax> <dgrid>  # GRIDS_METAD only: one line per metad CV
<t_min> <t_max>                    # t_max < 0  ->  use all COLVAR frames

# --- optional, after all windows ---
PRINT2D [AUTO_GRID]
  <icv2> [<gmin> <gmax> <dgrid>]   # 2nd CV for the 2D surface (not the restraint CV)
```

Rules:

- `<nwin>` is the first non-blank line. Blank lines and trailing `# ...`
  comments are ignored everywhere.
- `icv_mtd` and `icv2` are **1-based** CV indices.
- `GRIDS_METAD` is followed by exactly `ncv_mtd` grid lines (one per CV with
  `icv_metad == 1`), each `icv_mtd gmin gmax dgrid`.
- `AUTO_GRIDS_METAD` derives the metad grids from that window's COLVAR; no grid
  lines follow.
- **No-metad windows:** if a window's `plumed.dat` has no `METAD`, do **not**
  include a metad-grid mode line for it — write only `<folder>` followed by the
  `<t_min> <t_max>` line. The driver knows from `plumed.dat` whether to expect
  the grid-mode line.
- `PRINT2D` enables the 2D surface; the next line gives `icv2`. With `AUTO_GRID`
  on the `PRINT2D` line only `icv2` is read (grid auto-computed, pooled over all
  windows); otherwise supply `icv2 gmin gmax dgrid`. `icv2` must differ from the
  restraint CV.

**Auto-grid rule** (`AUTO_GRIDS_METAD`, `PRINT2D AUTO_GRID`):

```
gmin  = min(CV over the relevant COLVAR data)
gmax  = max(CV over the relevant COLVAR data)
dgrid = (gmax - gmin) / 50        # 50 bins  ->  51 grid points
```

If `gmax <= gmin` (e.g. all COLVAR values identical), the program **stops** with
an error instead of inventing a grid.

`AUTO_GRIDS_METAD` uses the single window's COLVAR; `PRINT2D AUTO_GRID` pools
over all windows so the 2D grid is common.

**Example** `tass.inp` (a metad window, a no-metad window, and a window with a
*different* metad CV):

```text
3                 # number of windows
W1
GRIDS_METAD
2 -3.3 -2.8 0.05  # metad on CV 2 (phi2): gmin gmax dgrid
1 -1              # t_min=1, t_max=all
W2
1 -1              # no metad in W2/plumed.dat -> no grid-mode line
W3
AUTO_GRIDS_METAD  # metad on a different CV; grid derived from W3/COLVAR
1 -1
PRINT2D AUTO_GRID
3                 # 2D surface vs CV 3 (psi2), auto grid
```

---

## 4. Command-line options

| Option | Default | Meaning |
|--------|---------|---------|
| `input` (positional) | `tass.inp` | Control file. |
| `--root DIR` | `.` | Base directory the window folders live under. |
| `--nranks N` | `1` | MPI ranks for the `vbias` step (needs `mpi4py` + `mpirun`). |
| `--threads T` | `1` | Threads per rank for `vbias`. |
| `--backend {numpy,numba}` | `numpy` | Shared-memory kernel for `vbias`. |
| `--force` | off | Recompute `ct.dat` / `vbias.dat` even if present. |
| `-v`, `--verbose` | off | Extra diagnostics. |

`ct.dat` and `vbias.dat` are reused if they already exist (override with
`--force`); the mean force, `fes_1d.dat`, and `free_energy_2D.dat` are always
rebuilt.

### Examples

```bash
# Serial, control file and windows in the current directory
python tass_reweight.py tass.inp

# Windows under ./data, 4 MPI ranks x 2 threads for vbias
python tass_reweight.py tass.inp --root ./data --nranks 4 --threads 2

# Threaded numba kernel and a full recompute
python tass_reweight.py tass.inp --backend numba --threads 8 --force
```

---

## 5. What the driver computes

For each window, with reweighting weight of MD frame `t`

```
w(t) = exp( ( V_bias(t) - c(t) ) / kT ),   kT = kB * TEMP  (the EXTENDED_LAGRANGIAN TEMP)
```

For a window **without** a `METAD` action, `V_bias = c(t) = 0`, so `w(t) = 1`
for every frame; `ct.dat` / `vbias.dat` are neither needed nor read (even with
`--force`). Steps 1–2 below are skipped for such windows.

1. **`c(t)`** (`ct.dat`, serial) — well-tempered MetaD time-dependent constant on
   the metad-CV grid; one value per HILLS row (`mtd_steps`).
2. **`V_bias(s,t)`** (`vbias.dat`, parallel) — bias felt at each MD frame, summed
   over hills deposited up to `mtd_max = (t * w_cv) // w_hill`; one value per
   COLVAR frame (`md_steps`), referenced to frame 1.
3. **Mean force** at the window's restraint centre `s1_at` (cv1 = restraint CV):

   ```
   dF1/ds1 = < -kappa_restraint * ( cv1(t) - s1_at ) >_reweighted
           = Σ_t w(t) [ -k (cv1(t) - s1_at) ] / Σ_t w(t)
   ```

   averaged over `t in [t_min, t_max]`; minimum-image (period `2*pi`) is applied
   to `cv1(t) - s1_at` when cv1 is `TORSION`.

Then, globally:

4. **`F1(s1)`** — windows are sorted by restraint centre and the mean force is
   integrated with the **trapezoidal rule** over the (possibly non-uniform)
   centres:

   ```
   F1(s_{k+1}) = F1(s_k) + 0.5 (s_{k+1} - s_k) ( dFds_k + dFds_{k+1} )
   ```

   (No `Pu.dat`, no spline fit — the gradient is the restraint mean force.)

5. **`F(cv1, cv2)`** (only if `PRINT2D`) — for each window a reweighted 1D
   histogram of cv2 over `[t_min, t_max]` gives `P_cond(cv2 | window)`; this is
   turned into a free energy and offset by that window's `F1`:

   ```
   F(cv1, cv2) = -kT ln P_cond(cv2 | window) + F1(cv1)
   ```

   cv2 is folded into `[gmin2, gmin2 + 2*pi)` when it is `TORSION`; each window
   contributes one cv1 row at its restraint centre.

---

## 6. Outputs

Per window (written inside the window folder; **metad windows only**):

| File | Columns |
|------|---------|
| `ct.dat` | `mtd_step  c(t)` |
| `vbias.dat` | `md_step  V_bias(t) - V_bias(1)` |

Global (written in the current working directory):

| File | Columns |
|------|---------|
| `mean_force.dat` | `s1   dF1/ds1   F1`  (one row per window, sorted by `s1`) |
| `fes_1d.dat` | `s1   F1(s1)` |
| `free_energy_2D.dat` | `cv1   cv2   F(cv1,cv2)   P_cond`  (blank line after each cv1 block) |

Plot (gnuplot):

```gnuplot
plot  'fes_1d.dat' u 1:2 w lp                  # 1D profile
splot 'free_energy_2D.dat' u 1:2:3 w pm3d      # 2D surface
```

---

## 7. Runtime behavior and parallelism

- The driver loops over windows **serially**; only `vbias` is parallelized.
- With `--nranks > 1` and `mpi4py` + `mpirun` available, the driver re-launches
  itself under `mpirun` once per window (a hidden worker mode) to compute
  `vbias.dat` across ranks; rank 0 writes the file. If MPI is unavailable it
  warns and falls back to a single threaded process.
- `--threads`/`--backend` control the shared-memory kernel within each rank
  (`numpy` vectorized, or a threaded `numba` kernel).

---

## 8. Troubleshooting

| Message / symptom | Likely cause |
|-------------------|--------------|
| `control file not found` | Wrong `input` path. |
| `plumed.dat not found for window K` | Folder name in `tass.inp` or `--root` is wrong. |
| `TASS expects exactly one RESTRAINT CV per window` | Zero or multiple `RESTRAINT` actions in that window's `plumed.dat`. |
| `icv2=... out of range 1..ncv` / `icv2 cannot be the restraint CV` | `PRINT2D` `icv2` invalid for one window's CV layout. |
| `TEMP not found on the EXTENDED_LAGRANGIAN line` | Add `TEMP=...` to `EXTENDED_LAGRANGIAN` (the TASS temperature). |
| `!!WARNING: TEMP keyword is missing along with METAD keywords` | Harmless — `METAD` `TEMP` is ignored; TASS uses the `EXTENDED_LAGRANGIAN` `TEMP`. |
| `expected GRIDS_METAD or AUTO_GRIDS_METAD ...` / wrong `t_min t_max` parse | A metad-grid line was given for a no-metad window, or omitted for a metad window. |
| `GRIDS_METAD missing grids for metad CV ...` | Fewer grid lines than metad CVs after `GRIDS_METAD`. |
| `the 2nd CV (icv2) cannot be the restraint CV` | Choose `icv2 != cv1`. |
| `two windows share the same restraint centre` | Two windows have identical `AT`. |
| `HILLS rows (...) != mtd_steps` | HILLS row count must match the number of hills. |
| `md_steps mismatch ... vbias.dat has ...` | Stale `vbias.dat`; rerun with `--force`. |
| `--nranks N requested but mpi4py/mpirun unavailable` | Falls back to single process; install MPI to parallelize. |

---

## 9. Quick start

```bash
# 1. Prepare W1/, W2/, ... each with plumed.dat, COLVAR, HILLS
# 2. Write tass.inp (Section 3.3)
# 3. Run
python tass_reweight.py tass.inp --nranks 4 --threads 2
# 4. Plot
gnuplot -e "plot 'fes_1d.dat' u 1:2 w lp; pause -1"
```
