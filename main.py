"""Set up and run simulations here."""

import os
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from agate.params import PhysicalParams
from agate.solver import solve

plt.style.use(["science", "nature", "grid"])

if not os.path.exists("data"):
    os.makedirs("data")

base = PhysicalParams("base", model_choice="full")
incompressible = PhysicalParams("incompressible", model_choice="incompressible")
ideal = PhysicalParams("ideal", model_choice="ideal")

simulations = []
for params in [base, incompressible, ideal]:
    params.bubble_radius = 1e-3
    simulations.append(solve(params.non_dimensionalise()))

z = np.linspace(-1, 0, 100)
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


plt.savefig("data/compressibility.pdf")
