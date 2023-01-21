"""Produce figures for paper"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from agate.params import PhysicalParams
from agate.solver import solve

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


def shade_regions(list_of_axes, height):
    for ax in list_of_axes:
        bottom = np.min(height)
        top = np.max(height)
        ax.axhspan(-1, bottom, facecolor="w", alpha=0)
        ax.axhspan(-1, 0, facecolor=GREY, alpha=0.2)
        ax.axhspan(0, top, facecolor="k", alpha=0.2)


if not os.path.exists("data"):
    os.makedirs("data")

"""fig 1 
Comparing:
full model with no gas (black)
full model with gas (green)
reduced model with gas (sand)."""
full_no_gas = PhysicalParams(
    name="full-no-gas", model_choice="full"
).non_dimensionalise()
full_no_gas.far_dissolved_concentration_scaled = 0
full_saturated = PhysicalParams(
    name="full-saturated", model_choice="full"
).non_dimensionalise()
reduced_saturated = PhysicalParams(
    name="reduced-saturated", model_choice="reduced"
).non_dimensionalise()

results = []
for parameters in [full_no_gas, full_saturated, reduced_saturated]:
    results.append(solve(parameters))

height = np.linspace(-2, 0.5, 1000)
colors = ["k", GREEN, SAND]
styles = ["solid", "dotted", "dashed"]

colors.reverse()
results.reverse()

fig = plt.figure(figsize=(5.5, 6.5), constrained_layout=True)
gs = fig.add_gridspec(ncols=3, nrows=2)
ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[0, 1], sharey=ax1)
ax3 = plt.subplot(gs[0, 2], sharey=ax1)
ax4 = plt.subplot(gs[1, 0])
ax5 = plt.subplot(gs[1, 1], sharey=ax4)
ax6 = plt.subplot(gs[1, 2], sharey=ax4)

for result, color, style in zip(results, colors, styles):
    kwargs = {"linestyle": style, "lw": 1.5}
    ax1.plot(result.temperature(height), height, color, label=result.name, **kwargs)
    ax2.plot(result.solid_fraction(height) * 100, height, color, **kwargs)
    ax3.plot(result.liquid_darcy_velocity(height), height, color, **kwargs)
    ax4.plot(result.concentration(height), height, color, **kwargs)
    ax5.plot(result.gas_fraction(height) * 100, height, color, **kwargs)
    ax6.plot(result.gas_density(height), height, color, **kwargs)

ax1.legend()
ax1.set_xlabel(r"Non dimensional temperature $\theta$")
ax2.set_xlabel(r"Solid fraction $\phi_s$ (\%)")
ax3.set_xlabel(r"Liquid Darcy velocity $W_l$")
ax1.set_ylabel(r"Scaled height $\eta$")
ax4.set_xlabel(r"Dissolved gas concentration $\omega$")
ax4.set_xlim(0.99, 1.1)
ax5.set_xlabel(r"Gas fraction $\phi_g$ (\%)")
ax6.set_xlabel(r"Gas density change $\psi$")
ax4.set_ylabel(r"Scaled height $\eta$")
shade_regions([ax2, ax1, ax3, ax4, ax5, ax6], height)
ax1.set_ylim(np.min(height), np.max(height))
ax4.set_ylim(np.min(height), np.max(height))
plt.setp(ax2.get_yticklabels(), visible=False)
plt.setp(ax3.get_yticklabels(), visible=False)
plt.setp(ax5.get_yticklabels(), visible=False)
plt.setp(ax6.get_yticklabels(), visible=False)
plt.savefig("data/base_results.pdf")
plt.close()


"""fig 4
plot showing convergence of high damkholer number to instant nucleation.
To push to higher damkholer number requires increasing max mesh nodes for solver"""
list_of_results = []
Da = []
supersaturation = []
colorbar = [
    "#FEE391",
    "#FEC44F",
    "#FB9A29",
    "#EC7014",
    "#CC4C02",
    "#993404",
    "#662506",
    "k",
]
for damkholer in [1, 5, 10, 50, 100, 500, 1000]:
    parameters = PhysicalParams(
        name=f"damkholer{damkholer:.0f}", model_choice="reduced"
    )
    parameters = parameters.non_dimensionalise()
    parameters.damkholer_number = damkholer
    list_of_results.append(solve(parameters, max_nodes=5000))
    Da.append(damkholer)
    supersaturation.append(np.max(list_of_results[-1].concentration_array - 1))

instant_nucleation = PhysicalParams(name="instant nucleation", model_choice="instant")
list_of_results.append(solve(instant_nucleation.non_dimensionalise()))
Da.append(np.inf)
supersaturation.append(0)

fig = plt.figure(figsize=(5, 5), constrained_layout=True)
gs = fig.add_gridspec(ncols=2, nrows=3)
ax1 = plt.subplot(gs[:2, 0])
ax2 = plt.subplot(gs[:2, 1], sharey=ax1)
ax3 = plt.subplot(gs[2, :])
for results, color in zip(list_of_results, colorbar):
    if results.params.model_choice == "reduced":
        ax1.plot(
            results.concentration(height),
            height,
            color,
            label=r"$\text{Da}=$" f" {results.params.damkholer_number:.0f}",
        )
    elif results.params.model_choice == "instant":
        ax1.plot(
            results.concentration(height), height, color, label=r"instant nucleation"
        )

    ax2.plot(results.gas_fraction(height) * 100, height, color)
    ax3.loglog(Da, supersaturation, "--*k")

ax1.set_xlabel(r"Dissolved gas concentration $\omega$")
ax2.set_xlabel(r"Gas fraction $\phi_g$ (\%)")
ax3.set_xlabel(r"Damkohler number $\text{Da}$")

ax1.set_ylabel(r"Scaled height $\eta$")
ax3.set_ylabel(r"Maximum supersaturation $\max{(\omega)} - 1$")

ax1.legend(loc="upper center", ncols=2)

ax1.set_ylim(-2, 1)
ax3.set_ylim(1e-3, 3)

ax1.set_xlim(0.9, 3.6)
ax2.set_xlim(-0.1, 3.1)
ax3.set_xlim(0.9, 1100)
shade_regions([ax1, ax2], height)
ax1.set_ylim(np.min(height), np.max(height))
plt.setp(ax2.get_yticklabels(), visible=False)
fig.suptitle(r"The effect of Damkohler number on gas bubble nucleation")
plt.savefig("data/nucleation_rate.pdf")
plt.close()


"""fig 5
Plotting the effect of changing bubble radius in instant nucleation model 
gas fraction for different bubble size
gas velocity for different bubble size
critical temperature for different bubble size"""
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
    temp = np.where(temp < -1, np.NaN, temp)
    temp = np.where(temp > 0, np.NaN, temp)
    return temp


fig = plt.figure(figsize=(5, 5), constrained_layout=True)
gs = fig.add_gridspec(ncols=2, nrows=3)
ax1 = plt.subplot(gs[:2, 0])
ax2 = plt.subplot(gs[:2, 1], sharey=ax1)
ax3 = plt.subplot(gs[2, :])
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

R_B = np.linspace(1e-5, 1e-3, 500)
ax3.plot(R_B * 1000, critical_temp(R_B), "k", zorder=0)
ax3.axhline(0)
ax3.axhline(-1)
ax1.set_xlabel(r"Gas fraction $\phi_g$ (\%)")
ax2.set_xlabel(r"Gas Darcy velocity $W_g$")
ax3.set_xlabel(r"Bubble radius $R_B$ (mm)")

ax1.set_ylabel(r"Scaled height $\eta$")
ax3.set_ylabel(r"Critical temperature $\theta_{\text{crit}}$")

ax2.legend(loc="lower center", ncols=2)

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
