from __future__ import annotations

import os
from typing import List, Tuple

import cupy as cp
import numpy as np




def mpi_context():
    try:
        from mpi4py import MPI  # type: ignore
    except Exception as e:
        raise RuntimeError("mpi4py is unavailable.") from e
    comm = MPI.COMM_WORLD
    return comm, comm.Get_rank(), comm.Get_size()


def bind_local_gpu() -> None:
    local_rank = int(
        os.environ.get(
            "SLURM_LOCALID",
            os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", os.environ.get("PMI_LOCAL_RANK", "0")),
        )
    )
    num_dev = int(cp.cuda.runtime.getDeviceCount())
    if num_dev > 0:
        cp.cuda.Device(local_rank % num_dev).use()
