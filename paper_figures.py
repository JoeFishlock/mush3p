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
height = np.linspace(-2, 1, 1000)

fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(ncols=2, nrows=1)
ax_left = plt.subplot(gs[0, 0])
ax_right = plt.subplot(gs[0, 1], sharey=ax_left)
ax_left.plot(results.temperature(height), height, "k")
ax_right.plot(results.solid_fraction(height) * 100, height, "k")
fig.suptitle(r"Simulation of steady mushy layer with no dissolved gas")
ax_left.set_xlabel(r"Non dimensional temperature $\theta$")
ax_right.set_xlabel(r"Solid fraction $\phi_s$ (\%)")
ax_left.set_ylabel(r"Scaled height $\eta$")
ax_left.set_xlim(-1.1, 0.1)
ax_left.set_ylim(-2, 1)
ax_right.set_xlim(-10, 110)
for ax in [ax_right, ax_left]:
    ax.fill_between([-100, 110], -1, -3, color="#004488", alpha=0.2)
    ax.fill_between([-100, 110], -1, 0, color="#DDAA33", alpha=0.2)
    ax.fill_between([-100, 110], 0, 1, color="#BB5566", alpha=0.2)

plt.setp(ax_right.get_yticklabels(), visible=False)
plt.savefig("data/base_results.pdf")


gas_flow = PhysicalParams(name="gas_flow", model_choice="full")
gas_flow = gas_flow.non_dimensionalise()
gas_flow.bubble_radius_scaled = 1
results = solve(gas_flow)

fig = plt.figure(figsize=(4.5, 3.5), constrained_layout=True)
gs = fig.add_gridspec(ncols=3, nrows=1)
ax_left = plt.subplot(gs[0, 0])
ax_right = plt.subplot(gs[0, 1], sharey=ax_left)
ax_end = plt.subplot(gs[0, 2], sharey=ax_left)
ax_left.plot(results.liquid_darcy_velocity(height), height, "k")
ax_right.plot(results.gas_fraction(height) * 100, height, "k")
ax_end.plot(results.concentration(height), height, "k")
fig.suptitle(r"Simulation of steady mushy layer containing dissolved gas")
ax_left.set_xlabel(r"Liquid Darcy velocity $W_l$")
ax_right.set_xlabel(r"Gas fraction $\phi_g$ (\%)")
ax_end.set_xlabel(r"Dissolved gas concentration $\omega$")
ax_left.set_ylabel(r"Scaled height $\eta$")
ax_left.set_xlim(-0.03, 0.005)
ax_left.set_ylim(-2, 1)
ax_right.set_xlim(-0.1, 3)
ax_end.set_xlim(0.99, 1.1)
for ax in [ax_right, ax_left, ax_end]:
    ax.fill_between([-100, 110], -1, -3, color="#004488", alpha=0.2)
    ax.fill_between([-100, 110], -1, 0, color="#DDAA33", alpha=0.2)
    ax.fill_between([-100, 110], 0, 1, color="#BB5566", alpha=0.2)

plt.setp(ax_right.get_yticklabels(), visible=False)
plt.setp(ax_end.get_yticklabels(), visible=False)
plt.savefig("data/gas_results.pdf")
