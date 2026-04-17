# Modified from EagarTsaiModel.py in ThermalControlLPBF-DRL repository.

import numpy as np
from scipy import signal
import scipy.integrate as integrate
from scipy.ndimage import gaussian_filter
from scipy import optimize
from scipy.ndimage import interpolation as intp
from scipy import interpolate as interp
import time
from scipy import special
import sys
from skimage import measure
import math
from matplotlib import pyplot as plt
from matplotlib import ticker


def _solve(xs, ys, zs, phi, coeff, rxf, rxr, ry, rz, D, V, sigma, dt):
    theta = np.ones((len(xs), len(ys), len(zs))) * 300
    integral_result = integrate.fixed_quad(
        _freefunc,
        dt / 50000,
        dt,
        args=(
            coeff,
            xs[:, None, None, None],
            ys[None, :, None, None],
            zs[None, None, :, None],
            phi,
            V,
            D,
            sigma,
            dt,
        ),
        n=75,
    )[0]
    theta += integral_result
    return theta


def _freefunc(x, coeff, x_coord, y, z, phi, V, D, sigma, dt):
    xp = -V * x * np.cos(phi)
    yp = -V * x * np.sin(phi)
    lmbda = np.sqrt(4 * D * x)
    gamma = np.sqrt(2 * sigma**2 + lmbda**2)
    start = (4 * D * x) ** (-3 / 2)

    termy = sigma * lmbda * np.sqrt(2 * np.pi) / (gamma)
    yexp1 = np.exp(-1 * ((y - yp) ** 2) / gamma**2)
    termx = termy
    xexp1 = np.exp(-1 * ((x_coord - xp) ** 2) / gamma**2)
    yintegral = termy * (yexp1)
    xintegral = termx * xexp1

    zintegral = 2 * np.exp(-(z**2) / (4 * D * x))
    value = coeff * start * xintegral * yintegral * zintegral
    return value


def _graft(theta, sol_theta, xs, ys, zs, l_idx, l_idy, l_new_x, l_new_y):
    y_offset = len(ys) // 2
    x_offset = len(xs) // 2
    x_roll = -x_offset + l_idx + l_new_x
    y_roll = -y_offset + l_idy + l_new_y
    theta += np.roll(sol_theta, (x_roll, y_roll, 0), axis=(0, 1, 2)) - 300
    return theta


class Solution:
    def __init__(self, dt, T0, phi, params):
        self.P = params["P"]
        self.V = params["V"]
        self.sigma = params["sigma"]
        self.A = params["A"]
        self.rho = params["rho"]
        self.cp = params["cp"]
        self.k = params["k"]
        self.D = self.k / (self.rho * self.cp)
        self.dimstep = params["dimstep"]
        self.xs = params["xs"]
        self.ys = params["ys"]
        self.zs = params["zs"]
        self.dt = dt
        self.T0 = T0
        self.a = 5
        self.phi = phi
        self.theta = np.ones((len(self.xs), len(self.ys), len(self.zs))) * self.T0

    def solve(self):
        coeff = self.P * self.A / (
            2 * np.pi * self.rho * self.cp * (self.sigma**2) * (np.pi) ** (3 / 2)
        )
        rxf = self.a * np.sqrt(self.sigma**2 + 2 * self.D * self.dt)
        rxr = self.a * np.sqrt(self.sigma**2 + 2 * self.D * self.dt) + self.V * self.dt
        ry = self.a * np.sqrt(self.sigma**2 + 2 * self.D * self.dt)
        rz = self.a * np.sqrt(2 * self.D * self.dt)
        self.theta = _solve(
            self.xs - self.xs[len(self.xs) // 2],
            self.ys - self.ys[len(self.ys) // 2],
            self.zs,
            self.phi,
            coeff,
            rxf,
            rxr,
            ry,
            rz,
            self.D,
            self.V,
            self.sigma,
            self.dt,
        )
        return self.theta

    def rotate(self):
        new_theta = np.ones((len(self.xs), len(self.ys), len(self.zs))) * self.T0
        orig_x = np.argmin(np.abs(self.xs))
        orig_y = np.argmin(np.abs(self.ys))
        origin = np.array([orig_x, orig_y])

        new_theta = np.roll(
            self.theta,
            (len(self.xs) // 2 - origin[0], len(self.ys) // 2 - origin[1]),
            axis=(0, 1),
        )
        rot_theta = intp.rotate(new_theta, angle=np.rad2deg(self.phi), reshape=False, cval=self.T0)
        new_theta = np.roll(
            rot_theta,
            (-len(self.xs) // 2 + origin[0], -len(self.ys) // 2 + origin[1]),
            axis=(0, 1),
        )
        self.theta = new_theta
        return self.theta

    def generate(self):
        return self.solve()


class EagarTsai:
    """
    Analytical E-T solution for heat transfer in laser powder bed fusion
    """

    def __init__(
        self,
        resolution,
        V=0.8,
        bc="flux",
        spacing=20e-6,
        x_extent=1000e-6,
        y_extent=1000e-6,
        depth=300e-6,
        *,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        init_location=(0.0, 0.0),
    ):
        self.P = 100
        self.V = V
        self.sigma = 50e-6 / 2 # 13.75e-6
        self.A = 0.3
        self.rho = 7910
        self.cp = 505
        self.k = 21.5
        self.bc = bc
        self.step = 0
        self.dimstep = resolution
        self.time = 0
        b = spacing
        # Backward-compatible defaults (legacy behavior used fixed -300µm minima).
        if x_min is None:
            x_min = -300e-6
        if x_max is None:
            x_max = x_extent
        if y_min is None:
            y_min = -300e-6
        if y_max is None:
            y_max = y_extent + b * 0.3

        self.xs = np.arange(float(x_min), float(x_max) + self.dimstep, step=self.dimstep)
        self.ys = np.arange(float(y_min), float(y_max) + self.dimstep, step=self.dimstep)
        self.zs = np.arange(-depth, 0 + self.dimstep, step=self.dimstep)

        self.theta = np.ones((len(self.xs), len(self.ys), len(self.zs))) * 300
        self.toggle = np.zeros((len(self.xs), len(self.ys)))
        self.D = self.k / (self.rho * self.cp)

        init_x, init_y = init_location
        self.location = [float(init_x), float(init_y)]
        self.location_idx = [
            int(np.argmin(np.abs(self.xs - self.location[0]))),
            int(np.argmin(np.abs(self.ys - self.location[1]))),
        ]
        self.a = 4
        self.times = []
        self.T0 = 300
        self.oldellipse = np.zeros((len(self.xs), len(self.ys)))
        self.store_idx = {}
        self.store = []
        self.visitedx = []
        self.visitedy = []
        self.state = None

    def forward_dummy(self, dt, phi, V=0.8, P=200):
        self.P = P
        self.V = V
        self.time += dt
        return self.theta

    def forward_diffuse_layer_truncated(self, dt, corrected, heat_source_list, k, n):
        if n == k:
            return heat_source_list[k]
        prev = corrected[n - 1]
        i = k - prev["k_start"]
        self.theta = prev["data"][i]
        return self.diffuse(dt)

    def forward_diffuse_layer(self, dt, corrected, heat_source_list, k, n):
        if n == k:
            return heat_source_list[k]
        self.theta = corrected[n - 1][k]
        return self.diffuse(dt)

    def merge_layer(self, local_corrected):
        self.theta = np.ones((len(self.xs), len(self.ys), len(self.zs))) * 300
        for term in local_corrected:
            self.theta += term - 300
        return self.theta

    def forward_heat(self, dt, phi, V=0.8, P=200):
        self.P = P
        self.V = V
        params = {
            "P": self.P,
            "V": V,
            "sigma": self.sigma,
            "A": self.A,
            "rho": self.rho,
            "cp": self.cp,
            "k": self.k,
            "dimstep": self.dimstep,
            "xs": self.xs,
            "ys": self.ys,
            "zs": self.zs,
        }

        self.state = "free"
        cache_key = (float(dt), float(phi), float(P), float(V))
        if cache_key in self.store_idx.keys():
            sol = self.store[self.store_idx[cache_key]]
        else:
            sol = Solution(dt, self.T0, phi, params)
            sol.generate()
            self.store_idx.update({cache_key: len(self.store)})
            self.store.append(sol)

        self.diffuse(sol.dt)
        self.graft(sol, phi)
        self.time += dt
        return self.theta

    def graft(self, sol, phi):
        l = sol.V * sol.dt
        l_new_x = int(np.rint(sol.V * sol.dt * np.cos(phi) / self.dimstep))
        l_new_y = int(np.rint(sol.V * sol.dt * np.sin(phi) / self.dimstep))
        self.theta = _graft(
            self.theta,
            sol.theta,
            sol.xs,
            sol.ys,
            sol.zs,
            self.location_idx[0],
            self.location_idx[1],
            l_new_x,
            l_new_y,
        )

        self.location[0] += l * np.cos(phi)
        self.location[1] += l * np.sin(phi)
        self.location_idx[0] = np.argmin(np.abs(self.location[0] - self.xs))
        self.location_idx[1] = np.argmin(np.abs(self.location[1] - self.ys))
        self.visitedx.append(self.location_idx[0])
        self.visitedy.append(self.location_idx[1])
        return self.theta

    def forward_diffuse_only(self, dt, phi, V=0.8, P=200):
        self.P = P
        self.V = V
        self.diffuse(dt)
        self.time += dt
        return self.theta

    def forward(self, dt, phi, V=0.8, P=200):
        self.P = P
        self.V = V
        params = {
            "P": self.P,
            "V": V,
            "sigma": self.sigma,
            "A": self.A,
            "rho": self.rho,
            "cp": self.cp,
            "k": self.k,
            "dimstep": self.dimstep,
            "xs": self.xs,
            "ys": self.ys,
            "zs": self.zs,
        }

        self.state = "free"
        cache_key = (float(dt), float(phi), float(P), float(V))
        if cache_key in self.store_idx.keys():
            sol = self.store[self.store_idx[cache_key]]
        else:
            sol = Solution(dt, self.T0, phi, params)
            sol.generate()
            self.store_idx.update({cache_key: len(self.store)})
            self.store.append(sol)

        self.diffuse(sol.dt)
        self.graft(sol, phi)
        self.time += dt
        return self.theta

    def diffuse(self, dt):
        diffuse_sigma = np.sqrt(2 * self.D * dt)
        padsize = int((4 * diffuse_sigma) // (self.dimstep * 2))
        if self.bc == "temp":
            if padsize == 0:
                padsize = 1
            theta_pad = np.pad(
                self.theta,
                ((padsize, padsize), (padsize, padsize), (padsize, padsize)),
                mode="reflect",
            ) - 300
            theta_pad_flip = np.copy(theta_pad)
            theta_pad_flip[-padsize:, :, :] = -theta_pad[-padsize:, :, :]
            theta_pad_flip[:padsize, :, :] = -theta_pad[:padsize, :, :]
            theta_pad_flip[:, -padsize:, :] = -theta_pad[:, -padsize:, :]
            theta_pad_flip[:, :padsize, :] = -theta_pad[:, :padsize, :]
            theta_pad_flip[:, :, :padsize] = -theta_pad[:, :, :padsize]
            theta_pad_flip[:, :, -padsize:] = theta_pad[:, :, -padsize:]
            theta_diffuse = (
                gaussian_filter(theta_pad_flip, sigma=diffuse_sigma / self.dimstep)[
                    padsize:-padsize, padsize:-padsize, padsize:-padsize
                ]
                + 300
            )
        if self.bc == "flux":
            if padsize == 0:
                padsize = 1
            theta_pad = np.pad(
                self.theta,
                ((padsize, padsize), (padsize, padsize), (padsize, padsize)),
                mode="reflect",
            ) - 300
            theta_pad_flip = np.copy(theta_pad)
            theta_pad_flip[-padsize:, :, :] = theta_pad[-padsize:, :, :]
            theta_pad_flip[:padsize, :, :] = theta_pad[:padsize, :, :]
            theta_pad_flip[:, -padsize:, :] = theta_pad[:, -padsize:, :]
            theta_pad_flip[:, :padsize, :] = theta_pad[:, :padsize, :]
            theta_pad_flip[:, :, :padsize] = -theta_pad[:, :, :padsize]
            theta_pad_flip[:, :, -padsize:] = theta_pad[:, :, -padsize:]
            theta_diffuse = (
                gaussian_filter(theta_pad_flip, sigma=diffuse_sigma / self.dimstep)[
                    padsize:-padsize, padsize:-padsize, padsize:-padsize
                ]
                + 300
            )
        self.theta = theta_diffuse
        return theta_diffuse

    def reset(self, *, init_location=(0.0, 0.0)):
        self.theta = np.ones((len(self.xs), len(self.ys), len(self.zs))) * self.T0
        init_x, init_y = init_location
        self.location = [float(init_x), float(init_y)]
        self.location_idx = [
            int(np.argmin(np.abs(self.xs - self.location[0]))),
            int(np.argmin(np.abs(self.ys - self.location[1]))),
        ]
        self.oldellipse = np.zeros((len(self.xs), len(self.ys)))
        self.store_idx = {}
        self.store = []
        self.visitedx = []
        self.visitedy = []
        self.state = None
        self.time = 0

    def func(self, x, h, y, z):
        coeff = self.A * self.P / (self.rho * self.cp * np.sqrt(self.D * 4 * np.pi**3))
        start = x ** (-0.5) / (self.sigma**2 + 2 * self.D * x)
        exponent = -1 * (
            ((h + self.V * x) ** 2 + y**2) / (2 * self.sigma**2 + 4 * self.D * x)
            + (z**2) / (4 * self.D * x)
        )
        value = coeff * np.exp(exponent) * start
        return value

    def get_coords(self):
        return self.xs, self.ys, self.zs

    def plot(self, vmax=1673):
        # Lazy import so core model can be imported without matplotlib/packaging.

        figures = []
        axes = []
        for _ in range(3):
            fig, ax = plt.subplots(figsize=(12, 12), dpi=200)
            figures.append(fig)
            axes.append(ax)
        xcurrent = np.argmax(self.theta[:, len(self.ys) // 2, -1])

        pcm0 = axes[0].pcolormesh(
            self.xs, self.ys, self.theta[:, :, -1].T, shading="gouraud", cmap="jet", vmin=300, vmax=vmax
        )
        pcm1 = axes[1].pcolormesh(
            self.xs,
            self.zs,
            self.theta[:, len(self.ys) // 2, :].T,
            shading="gouraud",
            cmap="jet",
            vmin=300,
            vmax=vmax,
        )
        pcm2 = axes[2].pcolormesh(
            self.ys,
            self.zs,
            self.theta[xcurrent, :, :].T,
            shading="gouraud",
            cmap="jet",
            vmin=300,
            vmax=vmax,
        )
        pcms = [pcm0, pcm1, pcm2]
        scale_x = 1e-6
        scale_y = 1e-6
        ticks_x = ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x / scale_x))
        ticks_y = ticker.FuncFormatter(lambda y, pos: "{0:g}".format(y / scale_y))
        titles = ["X - Y plane", "X - Z plane", "Y - Z plane"]
        axes[0].set_xlabel(r"x [$\mu$m]")
        axes[0].set_ylabel(r"y [$\mu$m]")
        axes[1].set_xlabel(r"x [$\mu$m]")
        axes[1].set_ylabel(r"z [$\mu$m]")
        axes[2].set_xlabel(r"y [$\mu$m]")
        axes[2].set_ylabel(r"z [$\mu$m]")

        for axis, pcm, fig, title in zip(axes, pcms, figures, titles):
            axis.set_aspect("equal")
            axis.xaxis.set_major_formatter(ticks_x)
            axis.yaxis.set_major_formatter(ticks_y)
            axis.set_title(
                str(round(self.time * 1e6))
                + r"[$\mu$s] "
                + "Power: "
                + str(int(np.around(self.P)))
                + "W"
                + " Velocity: "
                + str(np.around(self.V, decimals=2))
                + r" [m/s] "
                + title
            )
            clb = fig.colorbar(pcm, ax=axis)
            clb.ax.set_title(r"T [$K$]")
        return figures

    def meltpool(self, calc_length=False, calc_width=False):
        if calc_length:
            prop = measure.regionprops(np.array(self.theta[:, :, -1] > 1673, dtype="int"))
            prop_l = prop[0].major_axis_length * self.dimstep
            length = prop_l
        if calc_width:
            prop = measure.regionprops(np.array(self.theta[:, :, -1] > 1673, dtype="int"))
            prop_w = prop[0].minor_axis_length * self.dimstep
            width = prop_w
        depths = []
        for j in range(len(self.ys)):
            for i in range(len(self.xs)):
                if self.theta[i, j, -1] > 1673:
                    g = interp.CubicSpline(self.zs, self.theta[i, j, :] - 1673)
                    root = optimize.brentq(g, self.zs[0], self.zs[-1])
                    depths.append(root)
                    if root < self.toggle[i, j]:
                        self.toggle[i, j] = root
        if len(depths) == 0:
            depth = 0
        else:
            depth = np.min(depths)
        if calc_length and not calc_width:
            return length, depth
        if calc_width and not calc_length:
            return width, depth
        if calc_width and calc_length:
            return width, length, depth
        return depth

    def rotate(self, sol, phi):
        new_theta = np.copy(sol.theta)
        x_offset = len(self.xs) // 2
        y_offset = len(self.ys) // 2
        origin = np.array([x_offset, y_offset])
        new_theta = np.roll(
            new_theta,
            (len(self.xs) // 2 - origin[0], len(self.ys) // 2 - origin[1]),
            axis=(0, 1),
        )
        rot_theta = intp.rotate(new_theta, angle=np.rad2deg(phi), reshape=False, cval=self.T0)
        new_theta = np.roll(
            rot_theta,
            (-len(self.xs) // 2 + origin[0], -len(self.ys) // 2 + origin[1]),
            axis=(0, 1),
        )
        return new_theta
