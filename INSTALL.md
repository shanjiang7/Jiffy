# Installing JIFFY

## Prerequisites

- Linux with an NVIDIA GPU and a CUDA 12.x driver (all stages, including
  planning, initialize the GPU runtime).
- An MPI implementation for the multi-GPU experiments (validated with
  OpenMPI 5.0.5). Standard host-buffer MPI is sufficient; CUDA-aware MPI is
  **not** required.
- Python 3.11.
- On an HPC system: a module environment and Slurm (the provided job scripts
  use `sbatch`).

## 1. Load system modules

Module names are site-specific. On TACC Vista (the machine used for the
paper):

```bash
module purge
module load gcc/13.2.0
module load openmpi/5.0.5
module load cuda/12.5
```

On other systems, load any GCC toolchain, an MPI, and a CUDA 12.x toolkit.

## 2. Get the code

```bash
git clone https://github.com/shanjiang7/Jiffy.git
cd Jiffy
```

## 3. Create the Python environment

With Conda (recommended — all versions pinned):

```bash
conda env create -f environment.yml
conda activate hermes
```

Or with pip in a virtual environment. Python 3.11 must be the `python3` on
your PATH first — on TACC Vista the system python is 3.9, so load the
Python module into the stack (stepwise, in this order; the module
hierarchy rejects a single combined load):

```bash
module purge
module load gcc/13.2.0
module load python3/3.11.8
module load openmpi/5.0.5
module load cuda/12.5
```

```bash
python3 -m venv jiffy_env
source jiffy_env/bin/activate
pip install numpy==2.3.5 scipy==1.17.0
pip install cupy-cuda12x==13.6.0 numba==0.63.1
pip install mpi4py==4.1.1 networkx==3.6.1
pip install matplotlib pillow scikit-image
```

`mpi4py` builds against the system MPI, so load the MPI module (step 1)
before installing. If exact versions are unavailable on your system, the
closest versions compatible with CUDA 12.5 and Python 3.11 may be used.

## 4. Activate the environment

All runs are launched from the repository root with the environment script
sourced:

```bash
source env_vista.sh
```

The script loads the modules from step 1, sets the CUDA environment
(`CUDA_HOME`, `NUMBA_CUDA_DRIVER`), activates the `hermes` conda
environment, and puts `src/` on `PYTHONPATH`. If you created a virtual
environment with pip instead, activate it first — `env_vista.sh` detects
the active virtualenv, keeps it, and skips the conda activation:

```bash
source jiffy_env/bin/activate
source env_vista.sh
```

On systems other than TACC Vista, adapt the `module load` and CUDA lines,
and point `CONDA_ROOT` at your conda installation. The `PYTHONPATH` line
is location-independent.

## 5. Verify the installation

On a GPU node (e.g. an interactive session):

```bash
python3 -c "import cupy; cupy.zeros(1); print('GPU OK:', cupy.cuda.runtime.getDeviceCount(), 'device(s)')"
```

Then run a planning-only pass of the full pipeline (single GPU):

```bash
python3 -u src/hermes/scripts/segment_correction/plan_only.py \
  --config configs/examples/sim_ex1.ini \
  --path-config configs/examples/bull.ini \
  --dt-us 10 \
  --world-size 8 --planner-mode exact_dp
```

The first run builds the dependency-lookup table numerically (~80 s on a
GH200; cached in `.hermes_cache/` afterwards). With the cache in place the
whole command takes about half a minute.

A successful run prints the segment and component counts, the dependency-DAG
statistics, and the component-to-rank assignment with predicted per-rank
loads, and writes `planning_summary.json` under `outputs/segment_plan/`.

## 6. Multi-GPU notes

Rank-to-GPU binding reads the launcher's local-rank environment variable:
`SLURM_LOCALID`, falling back to `OMPI_COMM_WORLD_LOCAL_RANK`, then
`PMI_LOCAL_RANK`. Under Slurm (`srun`) this is automatic. Under a bare
`mpirun` from another MPI, make sure one of these variables is exported per
rank — if none is set, **every rank silently binds to GPU 0**.

The `r_eps` dependency-lookup table is cached in `.hermes_cache/` under the
**current working directory**; launch runs from the repository root so the
~80 s table build is paid only once.
