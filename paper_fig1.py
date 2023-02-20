"""Produce figures for paper"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import scienceplots
from agate.params import PhysicalParams
from agate.solver import solve, calculate_RMSE
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

fig = plt.figure(figsize=(5, 5), constrained_layout=True)
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
    if result.name != "full-no-gas":
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
    return (
        delta_gas_fraction,
        delta_gas_darcy_velocity,
        delta_solid_fraction,
        delta_mushy_layer_depth,
        delta_liquid_darcy_velocity,
        delta_frozen_gas_fraction,
        delta_concentration,
        delta_temperature,
        delta_gas_density,
    )


reduced_saturated, full_saturated, full_no_gas = results
"""(1) compare full_no_gas to full_saturated"""
print(calculate_simulation_difference(full_no_gas, full_saturated))
print(calculate_simulation_difference(reduced_saturated, full_saturated))
