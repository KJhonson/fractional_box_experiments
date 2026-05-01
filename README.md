# FEMX Experiments Clean

Clean repository for finite-element and fractional-operator experiments collected from:

- `FEM_project`: Python/FEniCSx 1D FEM/FVM fractional experiments and helpers.
- `error_analysis_lab/loop_experimental.R`: self-contained R 2D error-analysis loop that builds its mesh and transfer matrices before running the optimized sinc solver.

The original source folders were not modified.

## Repository Structure

```text
.
├── DESCRIPTION              # R project dependency metadata
├── README.md
├── requirements.txt         # Python dependency list
├── python_fem/
│   ├── 1d_experiments/      # Python 1D experiment scripts
│   ├── utils/               # Python helpers for loads, norms, sinc solves
│   └── *.py                 # Core FEM/FVM helper modules and validation scripts
├── R/
│   └── loop_experiment/
│       ├── loop_experimental.R
│       ├── sinc_solver_opt.R
│       ├── Basic_functions.R
│       └── Dualmesh(final+parallelize).R
└── outputs/
    └── .gitkeep
```

## Included Experiments

### Python FEM/FVM Experiments from `FEM_project`

The `python_fem/1d_experiments/` directory contains:

- `f_ex_xalpha.py`: fractional Dirichlet experiment with `f(x)=x^(-alpha)`.
- `f_ex_xalpha_apt.py`: related adaptive/approximation variant.
- `f_ex_indicator.py` and `f_ex_indicator_v2.py`: indicator-source fractional experiments.
- `f_ex_eigen.py` and `nf_ok_eigen.py`: eigenfunction-based experiments.
- `f_ok_indicator.py`: overkill/reference indicator experiment.
- `f_ok_stoch.py` and `stoch.nonuniform.py`: stochastic FVM experiments.
- `benckmark_fourier.py`: Fourier benchmark script.

Core Python helpers copied from `FEM_project` include `domains.py`, `operators.py`,
`playground.py`, `num_experiment.py`, `rational.py`, `rationalv2.py`,
`plot_visualization.py`, `jupyter_visualization.py`, `workflow_root.py`, and
`utils/`.

Generated `.dat` result folders, Python bytecode caches, logs, and system files were excluded.

### R Loop Experiment from `error_analysis_lab`

`R/loop_experiment/loop_experimental.R` is self-contained with respect to experiment state. It does not require a precomputed `checkpoint.RData`; instead it builds:

- `overkill_env` via `base_env(8, "LM", 2, kappa)`
- `approx_env` via `base_env(levels, "LM", 2, kappa)`
- dual-mesh transfer matrices `B` via `dm_evaluator(...)`
- interpolation matrices `A` via `fmesher::fm_evaluator(...)`

After building those objects, the script writes and reloads a local
`checkpoint.RData` during the run. That file is generated output and is ignored
by Git.

Direct local dependencies:

- `Basic_functions.R`: defines `base_env()` and mass-matrix helper routines.
- `Dualmesh(final+parallelize).R`: defines `dm_evaluator()` and dual-mesh utilities.
- `sinc_solver_opt.R`: defines dynamic sinc quadrature cache and assembly helpers.

## Setup

Detected local tool versions during cleanup:

- R 4.5.0
- Python 3.10.9

### R Packages

Required or inferred R packages:

- `Matrix`
- `future`
- `future.apply`
- `fmesher`
- `rSPDE`
- `splancs`
- `sf`
- `doSNOW`
- `foreach`

Install from R as needed:

```r
install.packages(c("Matrix", "future", "future.apply", "sf", "doSNOW", "foreach"))
```

`fmesher` and `rSPDE` may require their project-specific installation instructions.

### Python Packages

Required or inferred Python packages:

- `numpy`
- `scipy`
- `matplotlib`
- `mpmath`
- `mpi4py`
- `petsc4py`
- `dolfinx` / FEniCSx
- `ufl`
- `triangle`
- `IPython`
- `baryrat`

The FEniCSx stack (`dolfinx`, PETSc, MPI) is usually easiest to install via conda,
Docker, or a system-specific FEniCSx environment rather than plain `pip`.

## Running Experiments

Run commands from the repository root unless noted.

### R Loop Experiment

The R script sources helper files with relative paths, so run it from its own
directory:

```sh
cd R/loop_experiment
Rscript loop_experimental.R
```

Expected generated outputs:

- `R/loop_experiment/checkpoint.RData`
- `R/loop_experiment/errors2d.dat`
- `R/loop_experiment/betaxslope.dat`

These generated files are ignored by Git because they can be regenerated.

The script builds meshes and transfer matrices at runtime, then saves that state
to `checkpoint.RData`. This replaces the former dependency on a pre-existing
checkpoint and can be substantially more expensive than loading a saved workspace.

### Python 1D Experiments

The Python experiment scripts adjust `sys.path` relative to their file location and
expect the copied `utils/` package to be adjacent to `1d_experiments/`.

Examples:

```sh
python python_fem/1d_experiments/f_ex_xalpha.py
python python_fem/1d_experiments/f_ex_indicator_v2.py
python python_fem/1d_experiments/f_ok_stoch.py
python python_fem/1d_experiments/stoch.nonuniform.py
```

Several scripts write generated tables into a subdirectory named after the script,
for example `python_fem/1d_experiments/f_ex_xalpha/errors_1dfem.dat`. These
generated output directories are ignored by Git.

## Inputs and Outputs

No small standalone input data files were detected for the copied Python scripts.
Most inputs are parameters embedded in the experiment scripts.

The R loop does not require external input data. Its inputs are the parameters in
`loop_experimental.R`, including `kappa`, `levels`, and `betas`. Meshes and
matrices are generated at runtime by the included helper scripts. The generated
checkpoint is an intermediate output, not a source input.

Outputs are mostly `.dat` tables containing errors and fitted slopes. Previously
generated outputs from the messy source folders were not copied.

## Path Changes

No scientific logic was changed.

The copied scripts were kept in directory layouts that preserve their existing
relative path assumptions:

- Python sources are under `python_fem/`, with `1d_experiments/` and `utils/` as
  siblings.
- R loop files are colocated under `R/loop_experiment/`, matching the script's
  relative `source(...)` calls.

The R loop was changed from a precomputed-checkpoint form to a self-contained
form. It now calls `base_env()`, `dm_evaluator()`, and `fm_evaluator()` before
writing and reloading a runtime-generated `checkpoint.RData`. The previously
implicit dynamic-k beta was made explicit as `beta_for_k <- min(betas)` so the
precomputed sinc range is based on the most restrictive beta in the experiment
list.

## Reproducibility Notes

- The R loop now regenerates mesh and transfer-matrix state on each run before
  writing a local checkpoint.
- The Python experiments may be computationally expensive and require a working
  MPI/PETSc/FEniCSx environment.
- Validation performed during cleanup was limited to file listing, local reference
  checks, and parser checks. Full experiments were not run.

## Known Uncertainties

- The self-contained R loop may take significantly longer than the older
  precomputed-checkpoint version because it rebuilds `overkill_env`,
  `approx_env`, `A`, and `B`.
- Some Python files named like tests or playgrounds were retained because they
  import shared FEM helpers and may document validation workflows.
- `fmesher` and `rSPDE` installation sources were not inferred from the local files.
