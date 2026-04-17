"""matvec_cuda.py - nvcc-compiled CuPy RawKernel for the Level-3 Dirichlet matvec.

Replaces the Numba JIT version in the hot path.  Identical algorithm and
shared-memory tiling, but nvcc produces tighter PTX: better instruction
scheduling, register allocation, and memory-access hint emission than Numba's
LLVM-to-PTX path.

Public API
----------
call_mv_level3_dirichlet(blocks, threads, nx, ny, nz, v, result,
                          hixsq, hiysq, hizsq, dt05, n2, u0, hz)

    blocks, threads : (bx,by,bz), (tx,ty,tz) from launch_3d_xyz
    v, result       : cp.ndarray (float64, flat 1-D)
    remaining args  : Python float or int - converted to C types internally
"""
from __future__ import annotations

from string import Template
import numpy as np
import cupy as cp

# Default block dimensions.
_TX = 32
_TY = 4
_TZ = 4

# ---------------------------------------------------------------------------
# C CUDA kernel source
# ---------------------------------------------------------------------------

_KERNEL_SOURCE_TEMPLATE = Template(
    r"""
#define TX ${TX}
#define TY ${TY}
#define TZ ${TZ}

extern "C" __global__ void mv_level3_dirichlet_smem(
    int nx, int ny, int nz,
    const double* __restrict__ v,
    double* __restrict__ result,
    double hixsq, double hiysq, double hizsq,
    double dt05,
    double n2,
    double u0,
    double hz
) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;

    const int gi = blockIdx.x * TX + tx;   /* global x */
    const int gj = blockIdx.y * TY + ty;   /* global y */
    const int gk = blockIdx.z * TZ + tz;   /* global z */

    /* Shared-memory tile: smem[sz][sy][sx]
     *   sz = 0       : z-1 halo (gk-1 plane)
     *   sz = 1..TZ   : main data (tz = sz-1)
     *   sz = TZ+1    : z+1 halo (gk+1 plane)
     * Same convention for sy / y dimension.
     *
     * Layout is [TZ+2][TY+2][TX] so the fastest index (sx=tx) varies
     * contiguously within a warp - no shared-memory bank conflicts. */
    __shared__ double smem[TZ + 2][TY + 2][TX];

    const int nxny = nx * ny;

    /* ---- Phase 1: cooperative tile load ---- */

    /* Center element */
    smem[tz + 1][ty + 1][tx] = (gi < nx && gj < ny && gk < nz)
        ? v[gi + gj * nx + gk * nxny] : 0.0;

    /* Z-1 halo: threads on the bottom face (tz == 0) */
    if (tz == 0) {
        const int gk_m = gk - 1;
        smem[0][ty + 1][tx] = (gk_m >= 0 && gj < ny && gi < nx)
            ? v[gi + gj * nx + gk_m * nxny] : 0.0;
    }

    /* Z+1 halo: threads on the top face (tz == TZ-1) */
    if (tz == TZ - 1) {
        const int gk_p = gk + 1;
        smem[TZ + 1][ty + 1][tx] = (gk_p < nz && gj < ny && gi < nx)
            ? v[gi + gj * nx + gk_p * nxny] : 0.0;
    }

    /* Y-1 halo */
    if (ty == 0) {
        const int gj_m = gj - 1;
        smem[tz + 1][0][tx] = (gj_m >= 0 && gk < nz && gi < nx)
            ? v[gi + gj_m * nx + gk * nxny] : 0.0;
    }

    /* Y+1 halo */
    if (ty == TY - 1) {
        const int gj_p = gj + 1;
        smem[tz + 1][TY + 1][tx] = (gj_p < ny && gk < nz && gi < nx)
            ? v[gi + gj_p * nx + gk * nxny] : 0.0;
    }

    __syncthreads();

    /* ---- Phase 2: stencil compute ---- */

    if (gi >= nx || gj >= ny || gk >= nz) return;

    const int idxx = gi + gj * nx + gk * nxny;

    /* Boundary nodes: identity */
    if (gi == 0 || gi == nx - 1 || gj == 0 || gj == ny - 1 || gk == 0) {
        result[idxx] = v[idxx];
        return;
    }

    const double uC = smem[tz + 1][ty + 1][tx];
    const double uB = smem[tz    ][ty + 1][tx];   /* z-1 (smem) */
    const double uR = smem[tz + 1][ty + 2][tx];   /* y+1 (smem) */
    const double uL = smem[tz + 1][ty    ][tx];   /* y-1 (smem) */
    const double uI = v[idxx - 1];                 /* x-1 (global, coalesced) */
    const double uO = v[idxx + 1];                 /* x+1 (global, coalesced) */

    /* Robin BC at top surface: virtual z+1 neighbor */
    const double uT = (gk < nz - 1)
        ? smem[tz + 2][ty + 1][tx]          /* z+1 from smem */
        : uB - 2.0 * hz * n2 * uC;          /* Robin: uT = uB - 2*h*n2*uC */

    result[idxx] = -(uO + uI) * hixsq * dt05
                 - (uR + uL) * hiysq * dt05
                 - (uT + uB) * hizsq * dt05
                 + (2.0 * uC * (hixsq + hiysq + hizsq)) * dt05 + uC;
}

extern "C" __global__ void mv_level3_dirichlet_smem_dot(
    int nx, int ny, int nz,
    const double* __restrict__ v,
    double* __restrict__ result,
    double hixsq, double hiysq, double hizsq,
    double dt05,
    double n2,
    double u0,
    double hz,
    double* __restrict__ dot_out
) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    const int tid = tx + ty * TX + tz * TX * TY;

    const int gi = blockIdx.x * TX + tx;
    const int gj = blockIdx.y * TY + ty;
    const int gk = blockIdx.z * TZ + tz;
    const bool active = (gi < nx && gj < ny && gk < nz);

    __shared__ double smem[TZ + 2][TY + 2][TX];
    __shared__ double redbuf[TX * TY * TZ];

    const int nxny = nx * ny;

    smem[tz + 1][ty + 1][tx] = active
        ? v[gi + gj * nx + gk * nxny] : 0.0;

    if (tz == 0) {
        const int gk_m = gk - 1;
        smem[0][ty + 1][tx] = (gk_m >= 0 && gj < ny && gi < nx)
            ? v[gi + gj * nx + gk_m * nxny] : 0.0;
    }
    if (tz == TZ - 1) {
        const int gk_p = gk + 1;
        smem[TZ + 1][ty + 1][tx] = (gk_p < nz && gj < ny && gi < nx)
            ? v[gi + gj * nx + gk_p * nxny] : 0.0;
    }
    if (ty == 0) {
        const int gj_m = gj - 1;
        smem[tz + 1][0][tx] = (gj_m >= 0 && gk < nz && gi < nx)
            ? v[gi + gj_m * nx + gk * nxny] : 0.0;
    }
    if (ty == TY - 1) {
        const int gj_p = gj + 1;
        smem[tz + 1][TY + 1][tx] = (gj_p < ny && gk < nz && gi < nx)
            ? v[gi + gj_p * nx + gk * nxny] : 0.0;
    }

    __syncthreads();

    double contrib = 0.0;
    if (active) {
        const int idxx = gi + gj * nx + gk * nxny;
        const double uC = smem[tz + 1][ty + 1][tx];
        double ax_v;
        if (gi == 0 || gi == nx - 1 || gj == 0 || gj == ny - 1 || gk == 0) {
            ax_v = uC;
        } else {
            const double uB = smem[tz    ][ty + 1][tx];
            const double uR = smem[tz + 1][ty + 2][tx];
            const double uL = smem[tz + 1][ty    ][tx];
            const double uI = v[idxx - 1];
            const double uO = v[idxx + 1];
            const double uT = (gk < nz - 1)
                ? smem[tz + 2][ty + 1][tx]
                : uB - 2.0 * hz * n2 * uC;

            ax_v = -(uO + uI) * hixsq * dt05
                 - (uR + uL) * hiysq * dt05
                 - (uT + uB) * hizsq * dt05
                 + (2.0 * uC * (hixsq + hiysq + hizsq)) * dt05 + uC;
        }
        result[idxx] = ax_v;
        contrib = v[idxx] * ax_v;
    }

    redbuf[tid] = contrib;
    __syncthreads();

    for (int stride = (TX * TY * TZ) >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            redbuf[tid] += redbuf[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(dot_out, redbuf[0]);
    }
}

extern "C" __global__ void mv_level3_dirichlet_smem_dot_update_p(
    int nx, int ny, int nz,
    const double* __restrict__ r,
    const double* __restrict__ p_in,
    double* __restrict__ p_out,
    double* __restrict__ result,
    double hixsq, double hiysq, double hizsq,
    double dt05,
    double n2,
    double u0,
    double hz,
    double beta,
    double* __restrict__ dot_out
) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    const int tid = tx + ty * TX + tz * TX * TY;

    const int gi = blockIdx.x * TX + tx;
    const int gj = blockIdx.y * TY + ty;
    const int gk = blockIdx.z * TZ + tz;
    const bool active = (gi < nx && gj < ny && gk < nz);

    /* Smem X-dim extended by 2: center at sx=tx+1, halos at sx=0 and sx=TX+1.
     * All 6 stencil directions now served from smem; no global reads for p. */
    __shared__ double smem[TZ + 2][TY + 2][TX + 2];
    __shared__ double redbuf[TX * TY * TZ];

    const int nxny = nx * ny;
    double p_center = 0.0;

    /* Center: sx = tx+1 */
    if (active) {
        const int idxx = gi + gj * nx + gk * nxny;
        p_center = r[idxx] + beta * p_in[idxx];
        smem[tz + 1][ty + 1][tx + 1] = p_center;
    } else {
        smem[tz + 1][ty + 1][tx + 1] = 0.0;
    }

    /* Z-1 halo */
    if (tz == 0) {
        const int gk_m = gk - 1;
        smem[0][ty + 1][tx + 1] = (gk_m >= 0 && gj < ny && gi < nx)
            ? (r[gi + gj * nx + gk_m * nxny] + beta * p_in[gi + gj * nx + gk_m * nxny]) : 0.0;
    }
    /* Z+1 halo */
    if (tz == TZ - 1) {
        const int gk_p = gk + 1;
        smem[TZ + 1][ty + 1][tx + 1] = (gk_p < nz && gj < ny && gi < nx)
            ? (r[gi + gj * nx + gk_p * nxny] + beta * p_in[gi + gj * nx + gk_p * nxny]) : 0.0;
    }
    /* Y-1 halo */
    if (ty == 0) {
        const int gj_m = gj - 1;
        smem[tz + 1][0][tx + 1] = (gj_m >= 0 && gk < nz && gi < nx)
            ? (r[gi + gj_m * nx + gk * nxny] + beta * p_in[gi + gj_m * nx + gk * nxny]) : 0.0;
    }
    /* Y+1 halo */
    if (ty == TY - 1) {
        const int gj_p = gj + 1;
        smem[tz + 1][TY + 1][tx + 1] = (gj_p < ny && gk < nz && gi < nx)
            ? (r[gi + gj_p * nx + gk * nxny] + beta * p_in[gi + gj_p * nx + gk * nxny]) : 0.0;
    }
    /* X-1 halo: loaded by the leftmost thread in each X-row */
    if (tx == 0) {
        const int gi_m = gi - 1;
        smem[tz + 1][ty + 1][0] = (gi_m >= 0 && gj < ny && gk < nz)
            ? (r[gi_m + gj * nx + gk * nxny] + beta * p_in[gi_m + gj * nx + gk * nxny]) : 0.0;
    }
    /* X+1 halo: loaded by the rightmost thread in each X-row */
    if (tx == TX - 1) {
        const int gi_p = gi + 1;
        smem[tz + 1][ty + 1][TX + 1] = (gi_p < nx && gj < ny && gk < nz)
            ? (r[gi_p + gj * nx + gk * nxny] + beta * p_in[gi_p + gj * nx + gk * nxny]) : 0.0;
    }

    __syncthreads();

    double contrib = 0.0;
    if (active) {
        const int idxx = gi + gj * nx + gk * nxny;
        const double uC = smem[tz + 1][ty + 1][tx + 1];
        double ax_v;
        if (gi == 0 || gi == nx - 1 || gj == 0 || gj == ny - 1 || gk == 0) {
            ax_v = uC;
        } else {
            const double uB = smem[tz    ][ty + 1][tx + 1];
            const double uR = smem[tz + 1][ty + 2][tx + 1];
            const double uL = smem[tz + 1][ty    ][tx + 1];
            const double uI = smem[tz + 1][ty + 1][tx    ];  /* X-1 from smem */
            const double uO = smem[tz + 1][ty + 1][tx + 2];  /* X+1 from smem */
            const double uT = (gk < nz - 1)
                ? smem[tz + 2][ty + 1][tx + 1]
                : uB - 2.0 * hz * n2 * uC;

            ax_v = -(uO + uI) * hixsq * dt05
                 - (uR + uL) * hiysq * dt05
                 - (uT + uB) * hizsq * dt05
                 + (2.0 * uC * (hixsq + hiysq + hizsq)) * dt05 + uC;
        }
        result[idxx] = ax_v;
        p_out[idxx] = p_center;
        contrib = uC * ax_v;
    }

    redbuf[tid] = contrib;
    __syncthreads();

    for (int stride = (TX * TY * TZ) >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            redbuf[tid] += redbuf[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(dot_out, redbuf[0]);
    }
}

"""
)

_kernel_cache: dict[tuple[int, int, int], cp.RawKernel] = {}
_kernel_dot_cache: dict[tuple[int, int, int], cp.RawKernel] = {}
_kernel_dot_update_p_cache: dict[tuple[int, int, int], cp.RawKernel] = {}


def _build_kernel_source(tx: int, ty: int, tz: int) -> str:
    return _KERNEL_SOURCE_TEMPLATE.substitute(TX=tx, TY=ty, TZ=tz)


def _get_kernel(threads: tuple[int, int, int] = (_TX, _TY, _TZ)) -> cp.RawKernel:
    tx, ty, tz = threads
    key = (int(tx), int(ty), int(tz))
    kernel = _kernel_cache.get(key)
    if kernel is None:
        src = _build_kernel_source(*key)
        kernel = cp.RawKernel(src, "mv_level3_dirichlet_smem")
        _kernel_cache[key] = kernel
    return kernel


def _get_kernel_dot(threads: tuple[int, int, int] = (_TX, _TY, _TZ)) -> cp.RawKernel:
    tx, ty, tz = threads
    key = (int(tx), int(ty), int(tz))
    kernel = _kernel_dot_cache.get(key)
    if kernel is None:
        src = _build_kernel_source(*key)
        kernel = cp.RawKernel(src, "mv_level3_dirichlet_smem_dot")
        _kernel_dot_cache[key] = kernel
    return kernel


def _get_kernel_dot_update_p(threads: tuple[int, int, int] = (_TX, _TY, _TZ)) -> cp.RawKernel:
    tx, ty, tz = threads
    key = (int(tx), int(ty), int(tz))
    kernel = _kernel_dot_update_p_cache.get(key)
    if kernel is None:
        src = _build_kernel_source(*key)
        kernel = cp.RawKernel(src, "mv_level3_dirichlet_smem_dot_update_p")
        _kernel_dot_update_p_cache[key] = kernel
    return kernel


# ---------------------------------------------------------------------------
# Public call interface
# ---------------------------------------------------------------------------

def call_mv_level3_dirichlet(
    blocks, threads,
    nx, ny, nz,
    v: cp.ndarray,
    result: cp.ndarray,
    hixsq: float, hiysq: float, hizsq: float,
    dt05: float,
    n2: float,
    u0: float,
    hz: float,
) -> None:
    """Launch the nvcc-compiled Level-3 Dirichlet matvec kernel.

    Args match mv_level3_dirichlet_smem (Numba) exactly so the call site
    in segment_correction_outer_funcs.py is unchanged except the import.

    blocks : (bx, by, bz) - number of blocks from launch_3d_xyz
    threads: (tx, ty, tz) - threads per block from launch_3d_xyz
    """
    _get_kernel(threads)(
        blocks,
        threads,
        (
            np.int32(nx), np.int32(ny), np.int32(nz),
            v, result,
            np.float64(hixsq), np.float64(hiysq), np.float64(hizsq),
            np.float64(dt05), np.float64(n2), np.float64(u0), np.float64(hz),
        ),
    )


def call_mv_level3_dirichlet_with_dot(
    blocks, threads,
    nx, ny, nz,
    v: cp.ndarray,
    result: cp.ndarray,
    hixsq: float, hiysq: float, hizsq: float,
    dt05: float,
    n2: float,
    u0: float,
    hz: float,
    dot_out: cp.ndarray,
) -> None:
    """Launch matvec and accumulate dot(v, result) into dot_out[0]."""
    cp.cuda.runtime.memsetAsync(
        int(dot_out.data.ptr),
        0,
        int(dot_out.nbytes),
        cp.cuda.get_current_stream().ptr,
    )
    _get_kernel_dot(threads)(
        blocks,
        threads,
        (
            np.int32(nx), np.int32(ny), np.int32(nz),
            v, result,
            np.float64(hixsq), np.float64(hiysq), np.float64(hizsq),
            np.float64(dt05), np.float64(n2), np.float64(u0), np.float64(hz),
            dot_out,
        ),
    )


def call_mv_level3_dirichlet_with_dot_update_p(
    blocks, threads,
    nx, ny, nz,
    r: cp.ndarray,
    p_in: cp.ndarray,
    p_out: cp.ndarray,
    result: cp.ndarray,
    hixsq: float, hiysq: float, hizsq: float,
    dt05: float,
    n2: float,
    u0: float,
    hz: float,
    beta: float,
    dot_out: cp.ndarray,
) -> None:
    """Update p_out = r + beta*p_in, then matvec and dot(p_out, result)."""
    cp.cuda.runtime.memsetAsync(
        int(dot_out.data.ptr),
        0,
        int(dot_out.nbytes),
        cp.cuda.get_current_stream().ptr,
    )
    _get_kernel_dot_update_p(threads)(
        blocks,
        threads,
        (
            np.int32(nx), np.int32(ny), np.int32(nz),
            r, p_in, p_out, result,
            np.float64(hixsq), np.float64(hiysq), np.float64(hizsq),
            np.float64(dt05), np.float64(n2), np.float64(u0), np.float64(hz),
            np.float64(beta), dot_out,
        ),
    )


def call_mv_level3_dirichlet_tuned(
    nx: int,
    ny: int,
    nz: int,
    v: cp.ndarray,
    result: cp.ndarray,
    hixsq: float,
    hiysq: float,
    hizsq: float,
    dt05: float,
    n2: float,
    u0: float,
    hz: float,
    threads: tuple[int, int, int],
) -> None:
    """Launch a per-(tx,ty,tz)-compiled variant for tuning sweeps."""
    tx, ty, tz = threads
    blocks = (
        (int(nx) + tx - 1) // tx,
        (int(ny) + ty - 1) // ty,
        (int(nz) + tz - 1) // tz,
    )
    _get_kernel(threads)(
        blocks,
        threads,
        (
            np.int32(nx), np.int32(ny), np.int32(nz),
            v, result,
            np.float64(hixsq), np.float64(hiysq), np.float64(hizsq),
            np.float64(dt05), np.float64(n2), np.float64(u0), np.float64(hz),
        ),
    )
