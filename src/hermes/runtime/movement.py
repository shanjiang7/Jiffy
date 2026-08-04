import cupy as cp
from typing import Dict, Any
from hermes.kernels.interp import trilinear_interpolation 



__all__ = [
    "Index_3D_to_1D",
    "xy_index",
    "xy_index_negative",
    "grid_movement_index",
]

def Index_3D_to_1D(x_ind0, x_ind1, y_ind0, yind1, array):
    """
    Return a Fortran-order 1D view of array[x_ind0:x_ind1+1, y_ind0:yind1+1, 0:].
    """
    slice_3d = array[x_ind0:x_ind1+1, y_ind0:yind1+1, 0:]
    return slice_3d.ravel(order='F')

def xy_index(coord, velocity, t_len=None):
    """
    Find the first index where coord + velocity * cnt leaves [coord[0], coord[-1]].
    If t_len is provided, stop after t_len iterations (return None).
    """
    flag = True
    cnt_temp = 1
    while flag:
        temp_array = coord + velocity * cnt_temp
        interval_start = coord[0]
        interval_end   = coord[-1]
        outside = (temp_array < interval_start) | (temp_array > interval_end)

        if cp.any(outside):
            return int(cp.argmax(outside))
        cnt_temp += 1
        if t_len is not None and cnt_temp > t_len:
            return None

def xy_index_negative(coord, velocity, t_len=None):
    """
    Same as xy_index but steps in the negative direction: coord - velocity * cnt.
    """
    flag = True
    cnt_temp = 1
    while flag:
        temp_array = coord - velocity * cnt_temp
        interval_start = coord[0]
        interval_end   = coord[-1]
        outside = (temp_array < interval_start) | (temp_array > interval_end)

        if cp.any(outside):
            return int(cp.argmin(outside))
        cnt_temp += 1
        if t_len is not None and cnt_temp > t_len:
            return None

def update_after_movement_level3y_2(x, y, z, val, nx_old, ny_old, nz_old, hxval, hyval, hzval, xoldmin_lin, yoldmin_lin, zoldmin_lin,  nx, ny_in, nz, uin,  slice_1d_in, slice_1d_out, index_y, u0 , one,  val3, threads_per_block_in , blocks_per_grid_in):
    '''
    x,y,z: New coords after movement
    val: u_lin before update
    nx_old, ny_old, nz_old, hxval, hyval, hzval: Grid of val
    xoldmin_lin, yoldmin_lin, zoldmin_lin: Extracted from the coordinate of the level (before movement)
    nx, ny_in, nz: size of uin
    '''
    
    yin = y[0:index_y]

    trilinear_interpolation[blocks_per_grid_in, threads_per_block_in](x, yin, z, val, nx_old, ny_old, nz_old, hxval, hyval, hzval, xoldmin_lin, yoldmin_lin, zoldmin_lin, nx, ny_in, nz, uin, one)

    val3[slice_1d_in] = uin
    val3[slice_1d_out] = u0


def update_after_movement_level3y_2_negative(x, y, z, val, nx_old, ny_old, nz_old, hxval, hyval, hzval, xoldmin_lin, yoldmin_lin, zoldmin_lin,  nx, ny_in, nz, uin,  slice_1d_in, slice_1d_out, index_y, u0 , one,  val3, threads_per_block_in , blocks_per_grid_in):
    '''
    x,y,z: New coords after movement
    val: u_lin before update
    nx_old, ny_old, nz_old, hxval, hyval, hzval: Grid of val
    xoldmin_lin, yoldmin_lin, zoldmin_lin: Extracted from the coordinate of the level (before movement)
    ny, nx_in, nz: size of uin
    '''
    
    yin = y[index_y:]

    trilinear_interpolation[blocks_per_grid_in, threads_per_block_in](x, yin, z, val, nx_old, ny_old, nz_old, hxval, hyval, hzval, xoldmin_lin, yoldmin_lin, zoldmin_lin, nx, ny_in, nz, uin, one)

    val3[slice_1d_in] = uin
    val3[slice_1d_out] = u0

def update_after_movement_level3x_2(x, y, z, val, nx_old, ny_old, nz_old, hxval, hyval, hzval, xoldmin_lin, yoldmin_lin, zoldmin_lin,  nx_in, ny, nz, uin,  slice_1d_in, slice_1d_out, index_x, u0 , one,  val3, threads_per_block_in , blocks_per_grid_in):
    '''
    x,y,z: New coords after movement
    val: u_lin before update
    nx_old, ny_old, nz_old, hxval, hyval, hzval: Grid of val
    xoldmin_lin, yoldmin_lin, zoldmin_lin: Extracted from the coordinate of the level (before movement)
    ny, nx_in, nz: size of uin
    '''
    
    xin = x[0:index_x]

    trilinear_interpolation[blocks_per_grid_in, threads_per_block_in](xin, y, z, val, nx_old, ny_old, nz_old, hxval, hyval, hzval, xoldmin_lin, yoldmin_lin, zoldmin_lin, nx_in, ny, nz, uin, one)

    val3[slice_1d_in] = uin
    val3[slice_1d_out] = u0
    
def update_after_movement_level3x_2_negative(x, y, z, val, nx_old, ny_old, nz_old, hxval, hyval, hzval, xoldmin_lin, yoldmin_lin, zoldmin_lin,  nx_in, ny, nz, uin,  slice_1d_in, slice_1d_out, index_x, u0 , one,  val3, threads_per_block_in , blocks_per_grid_in):
    '''
    x,y,z: New coords after movement
    val: u_lin before update
    nx_old, ny_old, nz_old, hxval, hyval, hzval: Grid of val
    xoldmin_lin, yoldmin_lin, zoldmin_lin: Extracted from the coordinate of the level (before movement)
    ny, nx_in, nz: size of uin
    '''
    
    xin = x[index_x:]

    trilinear_interpolation[blocks_per_grid_in, threads_per_block_in](xin, y, z, val, nx_old, ny_old, nz_old, hxval, hyval, hzval, xoldmin_lin, yoldmin_lin, zoldmin_lin, nx_in, ny, nz, uin, one)

    val3[slice_1d_in] = uin
    val3[slice_1d_out] = u0

