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
plt.savefig("data/base_results.pdf")
plt.close()
