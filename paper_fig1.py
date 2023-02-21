"""Produce figures for paper"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import scienceplots
from tabulate import tabulate
from agate.params import PhysicalParams
from agate.solver import solve, calculate_RMSE
from agate.output import shade_regions

plt.style.use(["science", "ieee", "grid"])

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

fig = plt.figure(figsize=(6, 6), constrained_layout=True)
gs = fig.add_gridspec(ncols=3, nrows=2)
ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[0, 1], sharey=ax1)
ax3 = plt.subplot(gs[0, 2], sharey=ax1)
ax4 = plt.subplot(gs[1, 0])
ax5 = plt.subplot(gs[1, 1], sharey=ax4)
ax6 = plt.subplot(gs[1, 2], sharey=ax4)

for result in results:
    print(result.name)
    ax1.plot(result.temperature(height), height, label=result.name)
    ax2.plot(result.solid_fraction(height) * 100, height)
    ax3.plot(result.liquid_darcy_velocity(height), height)
    ax4.plot(result.concentration(height), height)
    ax5.plot(result.gas_fraction(height) * 100, height)
    if result.name != "full-no-gas":
        ax6.plot(result.gas_density(height), height)
    else:
        ax6.plot(np.NaN * result.gas_density(height), height)

ax1.legend(loc=3)
ax1.set_xlabel(r"temperature $\theta$")
ax2.set_xlabel(r"solid fraction $\phi_s$ (\%)")
ax3.set_xlabel(r"liquid Darcy velocity $W_l$")
ax1.set_ylabel(r"scaled height $\eta$")
ax4.set_xlabel(r"dissolved gas concentration $\omega$")
ax4.set_xlim(0.99, 1.1)
ax5.set_xlabel(r"gas fraction $\phi_g$ (\%)")
ax6.set_xlabel(r"gas density change $\psi$")
ax4.set_ylabel(r"scaled height $\eta$")
shade_regions([ax2, ax1, ax3, ax4, ax5, ax6], height)
ax1.set_ylim(np.min(height), np.max(height))
ax4.set_ylim(np.min(height), np.max(height))
plt.setp(ax2.get_yticklabels(), visible=False)
plt.setp(ax3.get_yticklabels(), visible=False)
plt.setp(ax5.get_yticklabels(), visible=False)
plt.setp(ax6.get_yticklabels(), visible=False)

"""Label the plots"""
for ax, label in zip(
    [ax1, ax2, ax3, ax4, ax5, ax6], ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
):
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
    ax.text(
        0.0,
        1.0,
        label,
        transform=ax.transAxes + trans,
        fontsize="medium",
        va="bottom",
        fontfamily="serif",
    )

plt.savefig("data/base_results.pdf")
plt.close()

"""calculate the RMSE error in each quantity for the pairs of models
ful-saturated compared with full-no-gas (1)
and reduced-saturated compared with full-saturated (2)

NOTE: gas velocity is zero for all of these simulations
"""


def calculate_simulation_difference(new, base):
    delta_concentration = calculate_RMSE(
        new.concentration_array,
        base.concentration_array,
        new.height_array,
        base.height_array,
    )
    delta_gas_fraction = calculate_RMSE(
        new.gas_fraction_array,
        base.gas_fraction_array,
        new.height_array,
        base.height_array,
    )
    delta_gas_darcy_velocity = calculate_RMSE(
        new.gas_darcy_velocity_array,
        base.gas_darcy_velocity_array,
        new.height_array,
        base.height_array,
    )
    delta_solid_fraction = calculate_RMSE(
        new.solid_fraction_array,
        base.solid_fraction_array,
        new.height_array,
        base.height_array,
    )
    delta_mushy_layer_depth = np.sqrt(
        (new.mushy_layer_depth - base.mushy_layer_depth) ** 2
        / (base.mushy_layer_depth) ** 2
    )
    delta_liquid_darcy_velocity = calculate_RMSE(
        new.liquid_darcy_velocity_array,
        base.liquid_darcy_velocity_array,
        new.height_array,
        base.height_array,
    )
    delta_frozen_gas_fraction = np.sqrt(
        (new.frozen_gas_fraction - base.frozen_gas_fraction) ** 2
        / (base.frozen_gas_fraction) ** 2
    )
    delta_temperature = calculate_RMSE(
        new.temperature_array,
        base.temperature_array,
        new.height_array,
        base.height_array,
    )
    delta_gas_density = calculate_RMSE(
        new.gas_density_array,
        base.gas_density_array,
        new.height_array,
        base.height_array,
    )
    report = {
        "comparison": f"{new.name} to {base.name}",
        "gas fraction": f"{delta_gas_fraction:.5f}",
        "gas Darcy velocity": f"{delta_gas_darcy_velocity:.5f}",
        "solid fraction": f"{delta_solid_fraction:.5f}",
        "mushy layer depth": f"{delta_mushy_layer_depth:.5f}",
        "liquid Darcy velocity": f"{delta_liquid_darcy_velocity:.5f}",
        "frozen gas fraction": f"{delta_frozen_gas_fraction:.5f}",
        "dissolved gas concentration": f"{delta_concentration:.5f}",
        "temperature": f"{delta_temperature:.5f}",
        "gas density": f"{delta_gas_density:.5f}",
    }
    return report


full_no_gas, full_saturated, reduced_saturated = results
"""Run comparisons and print table"""
report1 = calculate_simulation_difference(full_no_gas, full_saturated)
report2 = calculate_simulation_difference(reduced_saturated, full_saturated)
table = zip(report1.keys(), report1.values(), report2.values())
print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))
