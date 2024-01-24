# Hamiltonian Embedding

Hamiltonian embedding is a **hardware-efficient** approach to sparse Hamiltonian simulation that does not assume access to a black-box query model. This technique leverages both the *sparsity structure* of the input data and the *resource efficiency* of the underlying quantum hardware, enabling the deployment of interesting quantum applications on current quantum computers.

This is a joint work by [Jiaqi Leng](https://jiaqileng.github.io/), [Joseph Li](https://jli0108.github.io/), [Yuxiang Peng](https://pickspeng.github.io/) and [Xiaodi Wu](https://www.cs.umd.edu/~xwu/). The manuscript is available on [arXiv](https://arxiv.org/abs/2401.08550).


# Code organization
The source code is organized as follows:
- `src/experiments` contains scripts required for running the real-machine experiments.
It contains a file `ionq_circuit_utils.py` that handles the circuit compilation and sending jobs to IonQ.
In addition, the directory is subdivided into three subdirectories for each of the three computational tasks described in the paper.
    - `quantum_walk`
    - `spatial_search`
    - `real_space_sim`

These three directories contain a file `run_experiments.ipynb` which is used for running the IonQ experiments.
For the real-space simulation on QuEra, an additional notebook titled `QuEra_real_space_sim.ipynb` within the `real_space_sim/QuEra` directory.

- `src/resource_estimation` contains scripts required for the resource comparison between the conventional approach (i.e., the standard binary encoding) and Hamiltonian embeddings.
Within this directory, the files are divided as follows.
    - `scripts` provides the main scripts for running the resource estimation.
    - `data` contains all of estimated Trotter number and gate counts, stored as `.npz` files.
    - `plot` contains scripts for plotting the total gate counts.
    - `figures` stores the `.png` files for the resource analysis.

Finally, `src/fig_1a.py` is used to generate the matrix presented in Fig 1a, which represents the unary embedding of a 5-by-5 tridiagonal matrix.

# Usage

The code has been tested with Python 3.10 but should also work with some earlier versions such as 3.8 or 3.9.
There are several dependencies used in this project. Below are the relevant packages, along with the tested versions.
- numpy 1.23.5
- scipy 1.11.1
- networkx 3.2.1
- amazon-braket-default-simulator 1.18.3
- amazon-braket-schemas 1.19.0
- amazon-braket-sdk 1.51.0
- qiskit 0.44.1
- pytket 1.18.0

## Experiments

**Important**: In order to use the provided source code, it is necessary to create a file `.env` with an IonQ API key.
For example, if your API key is 00000000000000000000000000000000, the `.env` would contain the following.
```
IONQ_API_KEY=00000000000000000000000000000000
```

All Python scripts should be run from the project directory `hamiltonian-embedding`.

To run the experiments, Jupyter notebooks are provided for each task with the filename `run_experiments.ipynb`.
- `src/experiments/quantum_walk/run_experiments.ipynb`
- `src/experiments/spatial_search/run_experiments.ipynb`
- `src/experiments/real_space_sim/IonQ/run_experiments.ipynb`

These notebooks can be run without modification to reproduce the experimental results presented in the paper.

## Resource analysis

The empirical gate count comparison corresponding to each of the experiments (as listed in the tables of the main body) are computed using `src/resource_estimation/scripts/empirical_resource_comparison`, in which the parameters are chosen to be the same as those used for the real-machine experiments.
This notebook estimates the gate counts needed for standard binary to simulate the Hamiltonian to the same accuracy as in the experiments.
The gate counts for Hamiltonian embedding are directly taken from the circuits run in the experiments (i.e. found in the `run_experiments.ipynb` files).

For the systematic resource analysis of varying system sizes, we use the scripts in the directory `resource_estimation/scripts/resource_estimation_{name}.py`, where `{name}` is replaced by the name of the task.
To run these scripts without modification, it is highly suggested that they are run on an HPC cluster.
The typical runtime needed to run these scripts to completion is a few days (1-3 days depending on the task).
For small system sizes, the empirical resource comparison may be run on a laptop or PC, typically taking a few minutes.

After running these scripts, the data is saved in `resource_estimation/data`, and the scripts in `resource_estimation/plot` are used to generate the figures presented in the paper.
