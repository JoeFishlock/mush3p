"""Set up and run simulations here."""

from . import PhysicalParams, solve


def main():

    base = PhysicalParams("full", model_choice="full")
    incompressible = PhysicalParams("incompressible", model_choice="incompressible")
    reduced = PhysicalParams("reduced", model_choice="reduced")
    instant = PhysicalParams("instant_nucleation", model_choice="instant")

    simulations = []
    for params in [base, incompressible, reduced, instant]:
        params.bubble_radius = 1e-3
        simulations.append(solve(params.non_dimensionalise()))


if __name__ == "__main__":
    main()
