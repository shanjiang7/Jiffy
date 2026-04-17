def launch_1d(n_elems, threads=128):
    """Return (blocks, threads) for a 1D CUDA launch."""
    blocks = (n_elems + (threads - 1)) // threads
    return blocks, threads

def launch_3d(nx, ny, nz, threads=128):
    """Return (blocks, threads) for a flattened 3D domain."""
    return launch_1d(nx * ny * nz, threads)

def launch_3d_xyz(nx, ny, nz, tx=32, ty=4, tz=4):
    """Return 3D ((bx,by,bz), (tx,ty,tz)) launch config.

    Default 32×4×4 = 512 threads/block:
      - 32 in x aligns warps with the fastest-varying (i) dimension for coalescing.
      - 512 threads/block fills Blackwell SMs (2048 threads / 4 blocks = full occupancy).
    """
    bx = (nx + tx - 1) // tx
    by = (ny + ty - 1) // ty
    bz = (nz + tz - 1) // tz
    return (bx, by, bz), (tx, ty, tz)

def launch_bc(n_face, threads=128):
    """Return (blocks, threads) for face-sized launches (e.g. nx*nz, nx*ny)."""
    return launch_1d(n_face, threads)
