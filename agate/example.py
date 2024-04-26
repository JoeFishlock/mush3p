"""Set up and run simulations here."""

import os
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from agate.params import PhysicalParams
from agate.solver import solve

plt.style.use(["science", "nature", "grid"])


def main(data_path):
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    base = PhysicalParams("full", model_choice="full")
    incompressible = PhysicalParams("incompressible", model_choice="incompressible")
    ideal = PhysicalParams("ideal", model_choice="ideal")
    reduced = PhysicalParams("reduced", model_choice="reduced")
    instant = PhysicalParams("instant_nucleation", model_choice="instant")

    simulations = []
    for params in [base, incompressible, ideal, reduced, instant]:
        params.bubble_radius = 1e-3
        simulations.append(solve(params.non_dimensionalise()))

    z = np.linspace(-2, 1, 100)

    # PLotting gas fractions for different models
    plt.figure()
    for sim in simulations:
        plt.plot(
            sim.gas_fraction(z),
            z,
            label=f"{sim.name}",
        )
    plt.legend()
    plt.title(r"Gas fraction in mushy layer")
    plt.xlabel(r"$\phi_g$")
    plt.ylabel(r"scaled height $\eta$")

    plt.savefig(f"{data_path}/gas_fractions_for_different_models.pdf")

    def calculate_relative_error(true, new):
        diff = new - true
        error = diff * 100 / np.abs(true)
        return np.where(np.abs(true) > 1e-6, error, np.NaN)

    plt.figure()
    base_results = simulations[0]
    for sim in simulations[1:]:
        plt.plot(
            calculate_relative_error(base_results.gas_fraction(z), sim.gas_fraction(z)),
            z,
            label=f"{sim.name}",
        )
    plt.legend()
    plt.title("Difference in gas fraction from full model")
    plt.xlabel(r"Relative error \%")
    plt.ylabel(r"scaled height $\eta$")
    plt.savefig(f"{data_path}/gas_fraction_model_error.pdf")
    plt.close()


if __name__ == "__main__":
    DATA_PATH = "data"
    main(DATA_PATH)
