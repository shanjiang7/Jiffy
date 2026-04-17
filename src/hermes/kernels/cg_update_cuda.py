"""cg_update_cuda.py - CUDA kernels for fused CG vector updates."""
from __future__ import annotations

import numpy as np
import cupy as cp

_KERNEL_SOURCE = r"""
extern "C" __global__ void cg_update_xr_reduce_rr(
    int n,
    double alpha,
    double* __restrict__ x,
    double* __restrict__ r,
    const double* __restrict__ p,
    const double* __restrict__ ap,
    double* __restrict__ rr_out
) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int lane = threadIdx.x;
    __shared__ double redbuf[256];

    double contrib = 0.0;
    if (tid < n) {
        const double p_i = p[tid];
        const double ap_i = ap[tid];
        const double r_new = r[tid] - alpha * ap_i;
        x[tid] = x[tid] + alpha * p_i;
        r[tid] = r_new;
        contrib = r_new * r_new;
    }
    redbuf[lane] = contrib;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) redbuf[lane] += redbuf[lane + stride];
        __syncthreads();
    }
    if (lane == 0) atomicAdd(rr_out, redbuf[0]);
}
"""

_kernel_xr_rr: cp.RawKernel | None = None


def _get_kernel_xr_rr() -> cp.RawKernel:
    global _kernel_xr_rr
    if _kernel_xr_rr is None:
        _kernel_xr_rr = cp.RawKernel(_KERNEL_SOURCE, "cg_update_xr_reduce_rr")
    return _kernel_xr_rr


def _zero_device_scalar(buf: cp.ndarray) -> None:
    cp.cuda.runtime.memsetAsync(
        int(buf.data.ptr),
        0,
        int(buf.nbytes),
        cp.cuda.get_current_stream().ptr,
    )


def call_cg_update_xr_reduce_rr(
    x: cp.ndarray,
    r: cp.ndarray,
    p: cp.ndarray,
    ap: cp.ndarray,
    alpha: float,
    rr_out: cp.ndarray,
    threads: int = 256,
) -> None:
    n = int(x.size)
    blocks = (n + threads - 1) // threads
    _zero_device_scalar(rr_out)
    _get_kernel_xr_rr()(
        (blocks,),
        (threads,),
        (np.int32(n), np.float64(alpha), x, r, p, ap, rr_out),
    )
