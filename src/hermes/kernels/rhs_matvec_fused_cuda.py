"""rhs_matvec_fused_cuda.py — fused Level-3 RHS + first matvec init kernel.

Computes, in one pass over the grid:
  b = RHS(u, qs)
  Ax = A*u   (x0 is warm-start u)
  r = b - Ax
  p = r
"""
from __future__ import annotations

import numpy as np
import cupy as cp

_KERNEL_SOURCE = r"""
extern "C" __global__ void rhs_matvec_level3_fused_init(
    int nx, int ny, int nz,
    const double* __restrict__ u,
    const double* __restrict__ qs,
    double* __restrict__ b,
    double* __restrict__ r,
    double* __restrict__ p,
    double hixsq, double hiysq, double hizsq,
    double dt05,
    double n2, double n3,
    double u0,
    double hz,
    double* __restrict__ b_dot_out,
    double* __restrict__ rr_out
) {
    extern __shared__ double redbuf[];
    double* red_b = redbuf;
    double* red_rr = redbuf + blockDim.x;
    const int tid = threadIdx.x;

    const int idxx = blockDim.x * blockIdx.x + threadIdx.x;
    const int n_all = nx * ny * nz;
    double rhs_v = 0.0;
    double rr = 0.0;
    if (idxx < n_all) {
        const int i = idxx % nx;
        const int j = (idxx / nx) % ny;
        const int k = idxx / (nx * ny);

        const double uC = u[idxx];
        if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0) {
            rhs_v = u0;
            const double ax_v = uC;
            rr = rhs_v - ax_v;
            b[idxx] = rhs_v;
            r[idxx] = rr;
            p[idxx] = rr;
        } else {
            const int O = idxx + 1;
            const int I = idxx - 1;
            const int R = idxx + nx;
            const int L = idxx - nx;
            const int T = idxx + nx * ny;
            const int B = idxx - nx * ny;

            const double uI = u[I];
            const double uO = u[O];
            const double uR = u[R];
            const double uL = u[L];
            const double uB = u[B];
            const double uT_rhs = (k < nz - 1)
                ? u[T]
                : (uB + (4.0 * hz * (qs[i * ny + j] - n2 * n3)) - (2.0 * hz * n2 * uC));
            const double uT_mv = (k < nz - 1)
                ? u[T]
                : (uB - 2.0 * hz * n2 * uC);

            rhs_v =
                  (uI + uO) * hixsq * dt05
                + (uR + uL) * hiysq * dt05
                + (uT_rhs + uB) * hizsq * dt05
                + (-(2.0 * uC * (hixsq + hiysq + hizsq)) * dt05 + uC);

            const double ax_v =
                -(uO + uI) * hixsq * dt05
                -(uR + uL) * hiysq * dt05
                -(uT_mv + uB) * hizsq * dt05
                + (2.0 * uC * (hixsq + hiysq + hizsq)) * dt05
                + uC;

            rr = rhs_v - ax_v;
            b[idxx] = rhs_v;
            r[idxx] = rr;
            p[idxx] = rr;
        }
    }

    red_b[tid] = rhs_v * rhs_v;
    red_rr[tid] = rr * rr;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            red_b[tid] += red_b[tid + stride];
            red_rr[tid] += red_rr[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(b_dot_out, red_b[0]);
        atomicAdd(rr_out, red_rr[0]);
    }
}
"""

_kernel: cp.RawKernel | None = None


def _get_kernel() -> cp.RawKernel:
    global _kernel
    if _kernel is None:
        _kernel = cp.RawKernel(_KERNEL_SOURCE, "rhs_matvec_level3_fused_init")
    return _kernel


def call_rhs_matvec_level3_fused_init(
    nx: int,
    ny: int,
    nz: int,
    u: cp.ndarray,
    qs: cp.ndarray,
    b: cp.ndarray,
    r: cp.ndarray,
    p: cp.ndarray,
    hixsq: float,
    hiysq: float,
    hizsq: float,
    dt05: float,
    n2: float,
    n3: float,
    u0: float,
    hz: float,
    b_dot_out: cp.ndarray,
    rr_out: cp.ndarray,
    threads: int = 256,
) -> None:
    n_all = int(nx) * int(ny) * int(nz)
    blocks = (n_all + (threads - 1)) // threads
    stream_ptr = cp.cuda.get_current_stream().ptr
    cp.cuda.runtime.memsetAsync(int(b_dot_out.data.ptr), 0, int(b_dot_out.nbytes), stream_ptr)
    cp.cuda.runtime.memsetAsync(int(rr_out.data.ptr), 0, int(rr_out.nbytes), stream_ptr)
    _get_kernel()(
        (blocks,),
        (threads,),
        (
            np.int32(nx), np.int32(ny), np.int32(nz),
            u, qs, b, r, p,
            np.float64(hixsq), np.float64(hiysq), np.float64(hizsq),
            np.float64(dt05), np.float64(n2), np.float64(n3),
            np.float64(u0), np.float64(hz),
            b_dot_out, rr_out,
        ),
        shared_mem=2 * int(threads) * 8,
    )
