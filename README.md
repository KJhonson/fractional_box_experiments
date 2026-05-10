# Fractional Box Experiments

Numerical experiments for fractional elliptic operators using FEM/FVM discretizations.

## Structure

```text
.
├── python_fem/
│   ├── 1d_experiments/
│   └── utils/
├── R/
│   └── loop_experiment/
├── DESCRIPTION
├── requirements.txt
└── outputs/
```

## R Experiment

The main R experiment is:

```text
R/loop_experiment/loop_experimental.R
```

It builds the finite-element meshes and transfer matrices, runs the sinc quadrature solver for several values of `beta`, and writes error and slope tables.

Local R files:

- `Basic_functions.R`: mesh and FEM matrix construction.
- `Dualmesh(final+parallelize).R`: dual-mesh transfer operators.
- `sinc_solver_opt.R`: sinc quadrature routines.
- `loop_experimental.R`: experiment driver.

Run from the experiment directory:

```sh
cd R/loop_experiment
Rscript loop_experimental.R
```

Output files:

- `checkpoint.RData`
- `errors2d.dat`
- `betaxslope.dat`

## Python Experiments

The Python experiments are in:

```text
python_fem/1d_experiments/
```

Example commands:

```sh
python python_fem/1d_experiments/f_ex_xalpha.py
python python_fem/1d_experiments/f_ex_indicator_v2.py
python python_fem/1d_experiments/f_ok_stoch.py
python python_fem/1d_experiments/stoch.nonuniform.py
```

## Dependencies

R packages:

- `Matrix`
- `fmesher`
- `rSPDE`
- `future`
- `future.apply`
- `splancs`
- `sf`
- `doSNOW`
- `foreach`

Python packages:

- `numpy`
- `scipy`
- `matplotlib`
- `mpmath`
- `mpi4py`
- `petsc4py`
- `dolfinx`
- `ufl`
- `triangle`
- `IPython`
- `baryrat`

The Python experiments require a FEniCSx/PETSc/MPI environment.
