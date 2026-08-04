import numpy as np

__all__ = ["simu_parameter3d", "init_level3_outer"]  

class simu_parameter3d:
    """
    simulation params (outer/level-3).
    """
    def __init__(self, p, lxd, lyd, lzd, h_m, dt):
        
        # original dimensional length and non-dimensionalization
        self.lx = lxd / p.len_scale
        self.ly = lyd / p.len_scale
        self.lz = lzd / p.len_scale   
        self.h = h_m / p.len_scale
        self.dt = dt

        
        self.nx =  int(self.lx/self.h) + 1
        self.ny =  int(self.ly/self.h) + 1
        self.nz =  int(self.lz/self.h) + 1


        self.lz = (self.nz - 1) * self.h
        

        # self.t_end = 4* p.t_spot_on 
        # self.sol_max = int(( (self.t_end) / p.time_scale ) / self.dt) * 8

        #self.sol_max = max(int(sol_max_override), 1)
        # self.t_end = (self.sol_max - 1) * dt

        
        
        self.lx = (self.nx - 1) * self.h
        self.ly = (self.ny - 1) * self.h
        self.lz = (self.nz - 1) * self.h




def init_level3_outer(phys, float_type,  lxd, lyd, lzd, h_m, dt, xp):
    """
    Build the outer/level-3 arrays and constants.
    xp: numpy or cupy module (pass cupy as `xp` in GPU code).
    """
    # sol_max_override = (target_step + 1)
    sp = simu_parameter3d(phys, lxd, lyd, lzd, h_m, dt)

    # grid
    x_lin = xp.linspace(-sp.lx/2,  sp.lx/2,  num=sp.nx, dtype=float_type)
    y_lin = xp.linspace(-sp.ly/2,  sp.ly/2,  num=sp.ny, dtype=float_type)
    z_lin = xp.linspace(-sp.lz, 0,         num=sp.nz, dtype=float_type)
    x_lin0 = x_lin.copy()
    y_lin0 = y_lin.copy()
    z_lin0 = z_lin.copy()

    # time
    dt_lin = float_type(sp.dt)
    # t_vals_lin = xp.linspace(0, (sp.sol_max - 1) * dt_lin, sp.sol_max, dtype=float_type)

    # solver fields
    u0 = float_type((phys.T0 - phys.Ts) / phys.deltaT)
    size = sp.nx * sp.ny * sp.nz
    u_lin     = u0 * xp.ones(size, dtype=float_type)
    u_new_lin = u0 * xp.ones(size, dtype=float_type)
    b_lin     = xp.ones(size, dtype=float_type)

    # constants
    h_lin = float_type(sp.h)
    h_linisq  = float_type(1.0 / (h_lin**2))
    dt_lin05 = float_type(dt_lin * 0.5)

    out = dict(
        sp=sp,
        x_lin=x_lin, y_lin=y_lin, z_lin=z_lin,
        x_lin0=x_lin0, y_lin0=y_lin0, z_lin0=z_lin0,
        dt_lin=dt_lin,
        u_lin=u_lin, u_new_lin=u_new_lin, b_lin=b_lin,
        h_lin=h_lin, h_linisq=h_linisq, dt_lin05=dt_lin05,
        u0=u0,
    )
    return out
