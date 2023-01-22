import numpy as np
import matplotlib.pyplot as plt
from agate.params import PhysicalParams
from agate.solver import solve, calculate_RMSE
import scienceplots

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


def sensitivity_analysis(axes, parameter_string, parameter_range):
    base = PhysicalParams("base", model_choice="full")
    base = base.non_dimensionalise()
    base_result = solve(base)

    for num, value in enumerate(parameter_range):
        parameters = PhysicalParams(f"sim{num}", model_choice="full")
        parameters = parameters.non_dimensionalise()
        parameters.__dict__[parameter_string] = value
        result = solve(parameters)

        axes.plot(
            1,
            calculate_RMSE(
                result.concentration_array,
                base_result.concentration_array,
                result.height_array,
                base_result.height_array,
            ),
            TEAL,
            marker="*",
            linestyle=''
        )
        axes.plot(
            2,
            calculate_RMSE(
                result.temperature_array,
                base_result.temperature_array,
                result.height_array,
                base_result.height_array,
            ),
            ORANGE,
            marker="*",
            linestyle=''
        )
        axes.plot(
            3,
            calculate_RMSE(
                result.solid_fraction_array,
                base_result.solid_fraction_array,
                result.height_array,
                base_result.height_array,
            ),
            MAGENTA,
            marker="*",
            linestyle=''
        )
        axes.plot(
            4,
            calculate_RMSE(
                result.gas_fraction_array,
                base_result.gas_fraction_array,
                result.height_array,
                base_result.height_array,
            ),
            GREEN,
            marker="*",
            linestyle=''
        )
        axes.plot(
            5,
            calculate_RMSE(
                result.liquid_darcy_velocity_array,
                base_result.liquid_darcy_velocity_array,
                result.height_array,
                base_result.height_array,
            ),
            OLIVE,
            marker="*",
            linestyle=''
        )
        axes.plot(
            6,
            calculate_RMSE(
                result.gas_density_array,
                base_result.gas_density_array,
                result.height_array,
                base_result.height_array,
            ),
            SAND,
            marker="*",
            linestyle=''
        )
    # bottom, top = axes.get_ylim()
    # axes.set_ylim(0, max(top, 1))

fig = plt.figure(figsize=(4, 5), constrained_layout=True)
gs = fig.add_gridspec(nrows=5, ncols=1)
ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[1,0])
ax3 = plt.subplot(gs[2,0])
ax4 = plt.subplot(gs[3,0])
ax5 = plt.subplot(gs[4,0])
sensitivity_analysis(ax1, "gas_conductivity_ratio", np.linspace(0, 1, 5))
sensitivity_analysis(ax2, "damkholer_number", np.linspace(0, 70, 5))
sensitivity_analysis(ax3, "expansion_coefficient", np.geomspace(1e-4, 1e-2, 5))
sensitivity_analysis(ax4, "bubble_radius_scaled", np.geomspace(1e-4, 10, 10))
sensitivity_analysis(ax5, "stefan_number", np.linspace(1, 16, 10))
ax1.set_title("gas_conductivity_ratio")
ax2.set_title("damkholer_number")
ax3.set_title("expansion_coefficient")
ax4.set_title("bubble_radius_scaled")
ax5.set_title("stefan_number")
# TODO: Make them all share an x axis and use labels for the quantities 
# <22-01-23, Joe Fishlock> #
plt.savefig('data/sensitivity.pdf')
# TODO: group variables by gas or not to do with gas <22-01-23, Joe Fishlock> #
