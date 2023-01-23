import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from agate.params import PhysicalParams
from agate.solver import solve, calculate_RMSE
import scienceplots
import warnings

warnings.filterwarnings("ignore")

sns.set(rc={"text.usetex": True})
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

INPUTS = [
    "concentration_ratio",
    "stefan_number",
    "hele_shaw_permeability_scaled",
    "far_temperature_scaled",
    "damkholer_number",
    "expansion_coefficient",
    "stokes_rise_velocity_scaled",
    "bubble_radius_scaled",
    "far_dissolved_concentration_scaled",
    "gas_conductivity_ratio",
    "hydrostatic_pressure_scale",
    "laplace_pressure_scale",
    "kelvin_conversion_temperature",
    "atmospheric_pressure_scaled",
]

OUTPUTS = [
    "temperature_array",
    "solid_fraction_array",
    "liquid_darcy_velocity_array",
    "concentration_array",
    "gas_fraction_array",
    "gas_density_array",
    "gas_darcy_velocity_array",
    "mushy_layer_depth",
    "frozen_gas_fraction",
]

PARAMETER_RANGES = {
    "concentration_ratio": np.geomspace(0.1, 10, 5),
    "stefan_number": np.linspace(2, 16, 10),
    "hele_shaw_permeability_scaled": np.geomspace(1, 100, 7),
    "far_temperature_scaled": np.geomspace(0.01, 1, 7),
    "damkholer_number": np.linspace(0, 100, 10),
    "expansion_coefficient": np.geomspace(0.001, 0.1, 7),
    "stokes_rise_velocity_scaled": np.geomspace(1, 1e7, 10),
    "bubble_radius_scaled": np.geomspace(0.1, 1, 7),
    "far_dissolved_concentration_scaled": np.linspace(0, 1, 7),
    "gas_conductivity_ratio": np.linspace(0, 1, 5),
    "hydrostatic_pressure_scale": np.geomspace(1e-5, 1e-2, 5),
    "laplace_pressure_scale": np.geomspace(1e-5, 1e-1, 5),
    "kelvin_conversion_temperature": np.linspace(5, 20, 5),
    "atmospheric_pressure_scaled": np.geomspace(1e4, 1e8, 5),
}


def convert_input(input_name):
    conversion = {
        "concentration_ratio": r"$\mathcal{C}$",
        "stefan_number": r"$\mathcal{S}$",
        "hele_shaw_permeability_scaled": r"$\mathcal{K}$",
        "far_temperature_scaled": r"$\theta_\infty$",
        "damkholer_number": r"$\mathcal{D}$",
        "expansion_coefficient": r"$\chi$",
        "stokes_rise_velocity_scaled": r"$\mathcal{B}$",
        "bubble_radius_scaled": r"$\Lambda$",
        "far_dissolved_concentration_scaled": r"$\omega_\infty$",
        "gas_conductivity_ratio": r"$\nu_g$",
        "hydrostatic_pressure_scale": r"$\mathcal{H}$",
        "laplace_pressure_scale": r"$\text{La}$",
        "kelvin_conversion_temperature": r"$\theta_K$",
        "atmospheric_pressure_scaled": r"$p_0$",
    }
    return conversion[input_name]


def convert_output(output_name):
    conversion = {
        "temperature_array": r"$\Delta \theta$",
        "solid_fraction_array": r"$\Delta \phi_s$",
        "liquid_darcy_velocity_array": r"$\Delta W_l$",
        "concentration_array": r"$\Delta \omega$",
        "gas_fraction_array": r"$\Delta \phi_g$",
        "gas_density_array": r"$\Delta \rho_g$",
        "gas_darcy_velocity_array": r"$\Delta W_g$",
        "mushy_layer_depth": r"$\Delta z_L$",
        "frozen_gas_fraction": r"$\Delta \Psi$",
    }
    return conversion[output_name]


def calculate_sensitivity(
    input_parameter,
    input_range,
    output_parameter,
    base_non_dimensional_parameters,
):
    true_solution = solve(base_non_dimensional_parameters)
    differences = []
    for value in input_range:
        params = base_non_dimensional_parameters
        params.__dict__[input_parameter] = value
        solution = solve(params, max_nodes=5000)
        if (output_parameter == "mushy_layer_depth") or (
            output_parameter == "frozen_gas_fraction"
        ):
            numerator = (
                solution.__dict__[output_parameter]
                - true_solution.__dict__[output_parameter]
            )
            denominator = true_solution.__dict__[output_parameter]
            difference = np.sqrt((numerator / denominator) ** 2)

        elif (output_parameter == "gas_darcy_velocity") and (
            base_non_dimensional_parameters.bubble_radius_scaled >= 1
        ):
            difference = np.NaN

        elif (output_parameter == "liquid_darcy_velocity") and (
            base_non_dimensional_parameters.model_choice == "reduced"
        ):
            difference = np.NaN
        else:
            difference = calculate_RMSE(
                target_array=solution.__dict__[output_parameter],
                true_array=true_solution.__dict__[output_parameter],
                target_positions=solution.__dict__["height_array"],
                true_positions=true_solution.__dict__["height_array"],
            )
        differences.append(difference)
    differences = np.array(differences)
    return np.max(differences)


def calculate_table(base_non_dimensional_parameters):
    table = {}
    for input_name in tqdm(
        INPUTS, desc=f"sensitivity analysis for {base_non_dimensional_parameters.name}"
    ):
        outputs = {}
        for output_name in OUTPUTS:
            max_diff = calculate_sensitivity(
                input_name,
                PARAMETER_RANGES[input_name],
                output_name,
                base_non_dimensional_parameters,
            )
            outputs[convert_output(output_name)] = max_diff
        table[convert_input(input_name)] = outputs

    return table


def sensitivity_plot(axes, base_non_dimensional_parameters):
    df = pd.DataFrame(calculate_table(base_non_dimensional_parameters))
    sns.heatmap(df, annot=False, vmin=0, vmax=1, cmap="Reds", ax=axes)


base = PhysicalParams(name="base", model_choice="full")
base = base.non_dimensionalise()
base_mobile = PhysicalParams(name="base_mobile", model_choice="full")
base_mobile = base_mobile.non_dimensionalise()
base_mobile.bubble_radius_scaled = 0.1
base_reduced = PhysicalParams(name="base_reduced", model_choice="reduced")
base_reduced = base_reduced.non_dimensionalise()
base_mobile_reduced = PhysicalParams(name="base_mobile_reduced", model_choice="reduced")
base_mobile_reduced = base_mobile_reduced.non_dimensionalise()
base_mobile_reduced.bubble_radius_scaled = 0.1

fig = plt.figure(figsize=(8, 8), constrained_layout=True)
gs = fig.add_gridspec(ncols=2, nrows=2)
ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[0, 1])
ax3 = plt.subplot(gs[1, 0])
ax4 = plt.subplot(gs[1, 1])
sensitivity_plot(ax1, base)
sensitivity_plot(ax2, base_mobile)
sensitivity_plot(ax3, base_reduced)
sensitivity_plot(ax4, base_mobile_reduced)
ax1.set_title("Full model")
ax2.set_title("Full model with mobile gas phase")
ax3.set_title("Reduced model")
ax4.set_title("Reduced model with mobile gas phase")
plt.savefig("data/heatmap.pdf")
plt.close()

fig = plt.figure(figsize=(8, 8), constrained_layout=True)
ax1 = plt.gca()
sensitivity_plot(ax1, base_mobile)
ax1.set_title("Full model with mobile gas phase")
plt.savefig("data/heatmap_full.pdf")
plt.close()


# def sensitivity_analysis(axes, parameter_string, parameter_range):
#     base = PhysicalParams("base", model_choice="full")
#     base = base.non_dimensionalise()
#     base_result = solve(base)

#     for num, value in enumerate(parameter_range):
#         parameters = PhysicalParams(f"sim{num}", model_choice="full")
#         parameters = parameters.non_dimensionalise()
#         parameters.__dict__[parameter_string] = value
#         result = solve(parameters)

#         axes.plot(
#             1,
#             calculate_RMSE(
#                 result.concentration_array,
#                 base_result.concentration_array,
#                 result.height_array,
#                 base_result.height_array,
#             ),
#             TEAL,
#             marker="*",
#             linestyle="",
#         )
#         axes.plot(
#             2,
#             calculate_RMSE(
#                 result.temperature_array,
#                 base_result.temperature_array,
#                 result.height_array,
#                 base_result.height_array,
#             ),
#             ORANGE,
#             marker="*",
#             linestyle="",
#         )
#         axes.plot(
#             3,
#             calculate_RMSE(
#                 result.solid_fraction_array,
#                 base_result.solid_fraction_array,
#                 result.height_array,
#                 base_result.height_array,
#             ),
#             MAGENTA,
#             marker="*",
#             linestyle="",
#         )
#         axes.plot(
#             4,
#             calculate_RMSE(
#                 result.gas_fraction_array,
#                 base_result.gas_fraction_array,
#                 result.height_array,
#                 base_result.height_array,
#             ),
#             GREEN,
#             marker="*",
#             linestyle="",
#         )
#         axes.plot(
#             5,
#             calculate_RMSE(
#                 result.liquid_darcy_velocity_array,
#                 base_result.liquid_darcy_velocity_array,
#                 result.height_array,
#                 base_result.height_array,
#             ),
#             OLIVE,
#             marker="*",
#             linestyle="",
#         )
#         axes.plot(
#             6,
#             calculate_RMSE(
#                 result.gas_density_array,
#                 base_result.gas_density_array,
#                 result.height_array,
#                 base_result.height_array,
#             ),
#             SAND,
#             marker="*",
#             linestyle="",
#         )
#     # bottom, top = axes.get_ylim()
#     # axes.set_ylim(0, max(top, 1))


# fig = plt.figure(figsize=(4, 5), constrained_layout=True)
# gs = fig.add_gridspec(nrows=5, ncols=1)
# ax1 = plt.subplot(gs[0, 0])
# ax2 = plt.subplot(gs[1, 0])
# ax3 = plt.subplot(gs[2, 0])
# ax4 = plt.subplot(gs[3, 0])
# ax5 = plt.subplot(gs[4, 0])
# sensitivity_analysis(ax1, "gas_conductivity_ratio", np.linspace(0, 1, 5))
# sensitivity_analysis(ax2, "damkholer_number", np.linspace(0, 70, 5))
# sensitivity_analysis(ax3, "expansion_coefficient", np.geomspace(1e-4, 1e-2, 5))
# sensitivity_analysis(ax4, "bubble_radius_scaled", np.geomspace(1e-4, 10, 10))
# sensitivity_analysis(ax5, "stefan_number", np.linspace(1, 16, 10))
# ax1.set_title("gas_conductivity_ratio")
# ax2.set_title("damkholer_number")
# ax3.set_title("expansion_coefficient")
# ax4.set_title("bubble_radius_scaled")
# ax5.set_title("stefan_number")
# # TODO: Make them all share an x axis and use labels for the quantities
# # <22-01-23, Joe Fishlock> #
# plt.savefig("data/sensitivity.pdf")
# # TODO: group variables by gas or not to do with gas <22-01-23, Joe Fishlock> #
