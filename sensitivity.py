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
            outputs[output_name] = max_diff
        table[input_name] = outputs

    return table


def convert_table_to_matrix(table):
    inputs = []
    outputs = []
    for name in table["stefan_number"].keys():
        outputs.append(name)

    matrix = np.full((len(table.keys()), len(table["stefan_number"].keys())), np.NaN)
    for i, (input_name, pair) in enumerate(table.items()):
        inputs.append(input_name)
        for j, (_, value) in enumerate(pair.items()):
            matrix[i, j] = value
    return inputs, outputs, matrix


def sort_table(table):
    inputs, outputs, matrix = convert_table_to_matrix(table)
    row_indices = np.full((matrix.shape[0]), np.NaN)
    col_indices = np.full((matrix.shape[1]), np.NaN)
    row_indices = np.argsort(np.nansum(matrix, axis=1))[::-1]
    col_indices = np.argsort(np.nansum(matrix, axis=0))[::-1]
    sorted_matrix = np.full_like(matrix, np.NaN)
    for i, row_index in enumerate(row_indices):
        for j, col_index in enumerate(col_indices):
            sorted_matrix[i, j] = matrix[row_index, col_index]
    sorted_inputs = [inputs[i] for i in row_indices]
    sorted_outputs = [outputs[j] for j in col_indices]
    print(sorted_matrix)
    return sorted_inputs, sorted_outputs, sorted_matrix


def sensitivity_plot(axes, base_non_dimensional_parameters):
    table = calculate_table(base_non_dimensional_parameters)
    inputs, outputs, values = sort_table(table)
    for i, input_name in enumerate(inputs):
        inputs[i] = convert_input(input_name)

    for i, output_name in enumerate(outputs):
        outputs[i] = convert_output(output_name)

    sns.heatmap(
        values.T,
        vmin=0,
        vmax=1,
        square=True,
        cmap="Greys",
        ax=axes,
        annot=True,
        cbar=False,
        fmt=".1e",
        xticklabels=inputs,
        yticklabels=outputs,
    )


base_mobile = PhysicalParams(name="base_mobile", model_choice="full")
base_mobile = base_mobile.non_dimensionalise()
base_mobile.bubble_radius_scaled = 0.1

"""Sensitivity analysis of the full model with mobile gas phase"""
fig = plt.figure(figsize=(6, 6), constrained_layout=True)
ax1 = plt.gca()
sensitivity_plot(ax1, base_mobile)
plt.savefig("data/heatmap_full.pdf")
plt.close()
