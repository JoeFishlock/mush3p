"""Set up and run simulations here."""

import os
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from agate.params import PhysicalParams
from agate.solver import solve

plt.style.use(["science", "nature", "high-contrast", "grid"])

if not os.path.exists("data"):
    os.makedirs("data")

outputs = []
for i, bubble_radius in enumerate([1e-4, 5e-4, 1e-3]):
    params = PhysicalParams(f"RAD{i}", bubble_radius=bubble_radius)
    params.save(f"data/{params.name}")
    params_nd = params.non_dimensionalise()
    params_nd.save(f"data/{params_nd.name}_nd")
    outputs.append(solve(params_nd))


z = np.linspace(-1, 0, 100)
plt.figure()
for output in outputs:
    plt.plot(
        output.gas_fraction(z),
        z,
        label=f"{output.name}",
    )
plt.legend()
plt.title(r"Gas fraction in mushy layer")
plt.xlabel(r"$\phi_g$")
plt.ylabel(r"scaled height $\eta$")


plt.savefig("data/test.pdf")
