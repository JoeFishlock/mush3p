"""Produce figures for paper"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpltools import annotation
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

"""fig 4
plot showing convergence of high damkholer number to instant nucleation.
To push to higher damkholer number requires increasing max mesh nodes for solver"""
height = np.linspace(-2, 0.5, 1000)
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
    concentration = list_of_results[-1].concentration_array
    supersaturation.append(np.max(concentration - 1))

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
annotation.slope_marker((100, 0.01), (-1, 1), invert=True, ax=ax3, size_frac=0.2)

ax1.set_xlabel(r"Dissolved gas concentration $\omega$")
ax2.set_xlabel(r"Gas fraction $\phi_g$ (\%)")
ax3.set_xlabel(r"Damkohler number $\text{Da}$")

ax1.set_ylabel(r"Scaled height $\eta$")
ax3.set_ylabel(r"Maximum supersaturation")

ax1.legend(loc="upper center", ncols=2)
shade_regions([ax1, ax2], height)
ax1.set_ylim(np.min(height), np.max(height))
plt.setp(ax2.get_yticklabels(), visible=False)
fig.suptitle(r"The effect of Damkohler number on gas bubble nucleation")
plt.savefig("data/nucleation_rate.pdf")
plt.close()
