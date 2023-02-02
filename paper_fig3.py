"""Produce figures for paper"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from agate.params import PhysicalParams
from agate.solver import solve
from agate.output import shade_regions

plt.style.use(["science", "nature", "grid"])

GREEN = "#117733"
TEAL = "#44AA99"
ORANGE = "#EE7733"
RED = "#CC3311"
MAGENTA = "#EE3377"
HIGH_BLUE = "#004488"
HIGH_YELLOW = "#DDAA33"
HIGH_RED = "#BB5566"
GREY = "#BBBBBB"
OLIVE = "#999933"
SAND = "#DDCC77"


if not os.path.exists("data"):
    os.makedirs("data")

"""fig 5
Plotting the effect of changing bubble radius in instant nucleation model 
gas fraction for different bubble size
gas velocity for different bubble size
critical temperature for different bubble size"""
height = np.linspace(-2, 0.5, 1000)
list_of_results = []
sizes = []
colorbar = [
    GREY,
    "#FFF7BC",
    "#FEE391",
    "#FEC44F",
    "#FB9A29",
    "#EC7014",
    "#CC4C02",
    "#993404",
    "#662506",
    "k",
]
for bubble_radius in np.linspace(1e-4, 1e-3, 10):
    parameters = PhysicalParams(
        name=f"radius{bubble_radius:.0e}", model_choice="reduced"
    )
    throat_scale = parameters.reference_pore_scale
    parameters.bubble_radius = bubble_radius
    parameters = parameters.non_dimensionalise()
    list_of_results.append(solve(parameters))
    sizes.append(bubble_radius)


def critical_temp(R_B):
    lam = R_B / throat_scale

    temp = parameters.concentration_ratio * (1 - (1 / (lam**2)))
    temp = np.where(temp < -1, -1, temp)
    temp = np.where(temp > 0, 0, temp)
    return temp


fig = plt.figure(figsize=(5, 5), constrained_layout=True)
gs = fig.add_gridspec(ncols=2, nrows=3)
ax1 = plt.subplot(gs[:2, 0])
ax2 = plt.subplot(gs[:2, 1], sharey=ax1)
ax3 = plt.subplot(gs[2, 0])
ax4 = plt.subplot(gs[2, 1])
for results, color, size in zip(list_of_results, colorbar, sizes):
    ax1.plot(
        results.gas_fraction(height) * 100,
        height,
        color,
    )
    ax2.plot(
        results.gas_darcy_velocity(height),
        height,
        color,
        label=r"$R_B=$" f" {size*1000:.1f}mm",
    )
    ax3.plot(size * 1000, critical_temp(size), color, marker="*", linestyle="")

xaxis = np.linspace(0.01, 1.1, 30)
ax3.fill_between(xaxis, critical_temp(xaxis * 1e-3), -1, ec=None, fc=GREY)
ax3.fill_between(xaxis, 1, 0, ec="k", fc="none", linewidth=0.0, hatch="////")
ax3.fill_between(xaxis, -1, -2, ec="k", fc="none", linewidth=0.0, hatch="////")
R_B = np.linspace(1e-5, 1e-3, 500)
ax3.plot(R_B * 1000, critical_temp(R_B), "k", zorder=0)
ax3.axhline(0, linestyle="--", color="k")
ax3.axhline(-1, linestyle="--", color="k")
ax3.axvline(1, linestyle="dotted", color="k")

ax1.set_xlabel(r"Gas fraction $\phi_g$ (\%)")
ax2.set_xlabel(r"Gas Darcy velocity $W_g$")
ax3.set_xlabel(r"Bubble radius $R_B$ (mm)")

ax1.set_ylabel(r"Scaled height $\eta$")
ax3.set_ylabel(r"Temperature $\theta$")

h, l = ax2.get_legend_handles_labels()
ax4.legend(h, l, loc=10, ncols=2, edgecolor=(1, 1, 1, 1))
ax4.axis("off")

ax1.set_ylim(-2, 1)
ax3.set_ylim(-1.1, 0.1)

ax1.set_xlim(-0.1, 3.1)
ax2.set_xlim(-0.001, 0.026)
ax3.set_xlim(0.1, 1.1)
shade_regions([ax1, ax2], height)
ax1.set_ylim(np.min(height), np.max(height))
plt.setp(ax2.get_yticklabels(), visible=False)
fig.suptitle(r"The effect of bubble radius on gas transport")
plt.savefig("data/gas_cutoff.pdf")
plt.close()
