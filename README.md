# FEMX Experiments Clean

Clean repository for finite-element and fractional-operator experiments collected from:

- `FEM_project`: Python/FEniCSx 1D FEM/FVM fractional experiments and helpers.
- `error_analysis_lab/loop_experimental.R`: R 2D error-analysis loop using a saved checkpoint and optimized sinc solver.

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
│       ├── checkpoint.RData
│       ├── num_exp_deterministic.R
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

`R/loop_experiment/loop_experimental.R` was copied with its direct dependencies:

- `sinc_solver_opt.R`, sourced by `loop_experimental.R`.
- `checkpoint.RData`, loaded directly by `loop_experimental.R`.

The following files were also included because they appear to define or regenerate
objects stored in `checkpoint.RData`:

- `num_exp_deterministic.R`
- `Basic_functions.R`
- `Dualmesh(final+parallelize).R`

The checkpoint is large, but it is included because the loop script directly loads
it and does not run without it.

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

Install from R as needed:

```r
install.packages(c("Matrix", "future", "future.apply", "sf", "doSNOW"))
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

The R script uses relative paths to `checkpoint.RData` and `sinc_solver_opt.R`, so
run it from its own directory:

```sh
cd R/loop_experiment
Rscript loop_experimental.R
```

Expected generated outputs:

- `R/loop_experiment/errors2d.dat`
- `R/loop_experiment/betaxslope.dat`

These result tables are ignored by Git because they can be regenerated.

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

The R loop depends on `checkpoint.RData`, which contains precomputed mesh and matrix
objects such as `B`, `A`, `overkill_env`, `approx_env`, and `levels` according to
comments in the source script.

Outputs are mostly `.dat` tables containing errors and fitted slopes. Previously
generated outputs from the messy source folders were not copied.

## Path Changes

No scientific logic was changed.

The copied scripts were kept in directory layouts that preserve their existing
relative path assumptions:

- Python sources are under `python_fem/`, with `1d_experiments/` and `utils/` as
  siblings.
- R loop files are colocated under `R/loop_experiment/`, matching the original
  script's `load("checkpoint.RData")` and `source("sinc_solver_opt.R")` calls.

Because of this, no source path rewrites were required.

## Reproducibility Notes

- The R checkpoint was included because it is required by `loop_experimental.R`.
- `num_exp_deterministic.R` includes commented checkpoint-generation code. It was
  included as provenance for the checkpoint, but the exact saved state in
  `checkpoint.RData` should be treated as the authoritative input for the loop.
- The Python experiments may be computationally expensive and require a working
  MPI/PETSc/FEniCSx environment.
- Validation performed during cleanup was limited to file listing, local reference
  checks, and parser checks. Full experiments were not run.

## Known Uncertainties

- `checkpoint.RData` is a large generated artifact, but it is also a direct runtime
  dependency of `loop_experimental.R`.
- Some Python files named like tests or playgrounds were retained because they
  import shared FEM helpers and may document validation workflows.
- `fmesher` and `rSPDE` installation sources were not inferred from the local files.
