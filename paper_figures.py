"""Produce figures for paper"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from agate.params import PhysicalParams
from agate.solver import solve

plt.style.use(["science", "nature", "grid"])

if not os.path.exists("data"):
    os.makedirs("data")

base = PhysicalParams(name="BASE", model_choice="full")
base = base.non_dimensionalise()

base.bubble_radius_scaled = 1
base.far_dissolved_concentration_scaled = 0

results = solve(base)
height = np.linspace(-2, 1, 100)

# PLotting gas fractions for different models
fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(ncols=2, nrows=1)
ax_left = plt.subplot(gs[0, 0])
ax_right = plt.subplot(gs[0, 1], sharey=ax_left)
ax_left.plot(results.temperature(height), height, label=results.name)
ax_right.plot(results.solid_fraction(height)*100, height, label=results.name)
fig.suptitle(r"Simulation of steady mushy layer with no dissolved gas")
ax_left.set_xlabel(r"Non dimensional temperature $\theta$")
ax_right.set_xlabel(r"Solid fraction $\phi_s$ (\%)")
ax_left.set_ylabel(r"Scaled height $\eta$")
plt.setp(ax_right.get_yticklabels(), visible=False)


plt.savefig("data/base_results.pdf")
