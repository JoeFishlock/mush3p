"""Produce figures for paper"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from agate.params import PhysicalParams
from agate.solver import solve

plt.style.use(["science", "nature", "grid"])
GREEN = "#228833"
TEAL = "#009988"
ORANGE = "#EE7733"
RED = "#CC3311"
MAGENTA = "#EE3377"
HIGH_BLUE = "#004488"
HIGH_YELLOW = "#DDAA33"
HIGH_RED = "#BB5566"
GREY = "#BBBBBB"


def shade_regions(list_of_axes):
    for ax in list_of_axes:
        ax.fill_between([-100, 110], -1, -3, color="w", alpha=0)
        ax.fill_between([-100, 110], -1, 0, color=GREY, alpha=0.2)
        ax.fill_between([-100, 110], 0, 1, color="k", alpha=0.2)


if not os.path.exists("data"):
    os.makedirs("data")

"""fig 1 
full model comparison of gas and no gas for temperature, solid fraction and liquid
darcy velocity."""
base = PhysicalParams(name="no gas", model_choice="full")
base = base.non_dimensionalise()
base.far_dissolved_concentration_scaled = 0
results = solve(base)

gas_flow = PhysicalParams(name="saturated", model_choice="full")
gas_flow = gas_flow.non_dimensionalise()
results_gas = solve(gas_flow)

height = np.linspace(-2, 1, 1000)

fig = plt.figure(figsize=(4.5, 3.5), constrained_layout=True)
gs = fig.add_gridspec(ncols=3, nrows=1)
ax_left = plt.subplot(gs[0, 0])
ax_right = plt.subplot(gs[0, 1], sharey=ax_left)
ax_end = plt.subplot(gs[0, 2], sharey=ax_left)
ax_left.plot(results_gas.temperature(height), height, GREEN, label=results_gas.name)
ax_left.plot(results.temperature(height), height, "k", label=results.name)
ax_left.legend()
ax_right.plot(results_gas.solid_fraction(height) * 100, height, GREEN)
ax_right.plot(results.solid_fraction(height) * 100, height, "k")
ax_end.plot(results_gas.liquid_darcy_velocity(height), height, GREEN)
ax_end.plot(results.liquid_darcy_velocity(height), height, "k")
fig.suptitle(r"Simulation of steady mushy layer with no dissolved gas")
ax_left.set_xlabel(r"Non dimensional temperature $\theta$")
ax_right.set_xlabel(r"Solid fraction $\phi_s$ (\%)")
ax_end.set_xlabel(r"Liquid Darcy velocity $W_l$")
ax_left.set_ylabel(r"Scaled height $\eta$")
ax_left.set_xlim(-1.1, 0.1)
ax_left.set_ylim(-2, 1)
ax_right.set_xlim(-10, 110)
ax_end.set_xlim(-0.03, 0.005)
shade_regions([ax_right, ax_left, ax_end])
plt.setp(ax_right.get_yticklabels(), visible=False)
plt.setp(ax_end.get_yticklabels(), visible=False)
plt.savefig("data/base_results.pdf")
plt.close()

"""fig 2
full model default behaviour for gas fraction and concentration."""
fig = plt.figure(figsize=(4.5, 3.5), constrained_layout=True)
gs = fig.add_gridspec(ncols=3, nrows=1)
ax_left = plt.subplot(gs[0, 0])
ax_right = plt.subplot(gs[0, 1], sharey=ax_left)
ax_end = plt.subplot(gs[0, 2], sharey=ax_left)
ax_left.plot(results_gas.concentration(height), height, GREEN)
ax_right.plot(results_gas.gas_fraction(height) * 100, height, GREEN)
ax_end.plot(results.gas_density(height), height, GREEN)
fig.suptitle(r"Simulation of steady mushy layer containing dissolved gas")
ax_left.set_xlabel(r"Dissolved gas concentration $\omega$")
ax_right.set_xlabel(r"Gas fraction $\phi_g$ (\%)")
ax_end.set_xlabel(r"Gas density change $\psi$")
ax_left.set_ylabel(r"Scaled height $\eta$")
ax_left.set_xlim(0.99, 1.1)
ax_left.set_ylim(-2, 1)
ax_right.set_xlim(-0.1, 3)
ax_end.set_xlim(1.0, 1.10)
shade_regions([ax_right, ax_left, ax_end])
plt.setp(ax_right.get_yticklabels(), visible=False)
plt.setp(ax_end.get_yticklabels(), visible=False)
plt.savefig("data/gas_results.pdf")
plt.close()

"""fig 3
comparison of the different models for
solid fraction, temperature, gas fraction and liquid velocity
for default parameters so no gas buoyant rise."""
full = PhysicalParams(name="full", model_choice="full")
incompressible = PhysicalParams(name="incompressible", model_choice="incompressible")
thermally_ideal = PhysicalParams(name="thermally ideal", model_choice="ideal")
reduced = PhysicalParams(name="reduced", model_choice="reduced")
instant_nucleation = PhysicalParams(name="instant nucleation", model_choice="instant")

list_of_results = []
for parameters in [full, incompressible, thermally_ideal, reduced, instant_nucleation]:
    parameters = parameters.non_dimensionalise()
    list_of_results.append(solve(parameters))

fig = plt.figure(figsize=(5, 5), constrained_layout=True)
gs = fig.add_gridspec(ncols=2, nrows=2)
ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[0, 1], sharey=ax1)
ax3 = plt.subplot(gs[1, 0])
ax4 = plt.subplot(gs[1, 1], sharey=ax3)

for results, color in zip(list_of_results, ["k", "--b", "g", "--m", "--r"]):
    ax1.plot(results.temperature(height), height, color, label=results.name)
    ax2.plot(results.solid_fraction(height) * 100, height, color, label=results.name)
    ax3.plot(results.liquid_darcy_velocity(height), height, color, label=results.name)
    ax4.plot(results.gas_fraction(height) * 100, height, color, label=results.name)

fig.suptitle(r"Comparison of different models")
ax1.legend()
ax1.set_xlabel(r"Temperature $\theta$")
ax2.set_xlabel(r"Solid fraction $\phi_s$ (\%)")
ax3.set_xlabel(r"Liquid Darcy velocity $W_l$")
ax4.set_xlabel(r"Gas fraction $\phi_g$ (\%)")
ax1.set_ylabel(r"Scaled height $\eta$")
ax3.set_ylabel(r"Scaled height $\eta$")
ax1.set_xlim(-1.1, 0.1)
ax2.set_xlim(-10, 110)
ax3.set_xlim(-0.03, 0.005)
ax4.set_xlim(-0.1, 3)
ax1.set_ylim(-2, 1)
ax3.set_ylim(-2, 1)
shade_regions([ax1, ax2, ax3, ax4])
plt.setp(ax2.get_yticklabels(), visible=False)
plt.setp(ax4.get_yticklabels(), visible=False)
plt.savefig("data/model_comparison.pdf")
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
shade_regions([ax1, ax2])
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
shade_regions([ax1, ax2])
plt.setp(ax2.get_yticklabels(), visible=False)
fig.suptitle(r"The effect of bubble radius on gas transport")
plt.savefig("data/gas_cutoff.pdf")
plt.close()
