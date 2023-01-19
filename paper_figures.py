"""Produce figures for paper"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from agate.params import PhysicalParams
from agate.solver import solve

plt.style.use(["science", "nature", "grid"])
GREEN = "#228833"
TEAL = "#009988"
ORANGE = "#EE7733"
RED = "#CC3311"
MAGENTA = "#EE3377"

if not os.path.exists("data"):
    os.makedirs("data")

"""fig 1 
full model comparison of gas and no gas for temperature, solid fraction and liquid
darcy velocity."""
base = PhysicalParams(name="no gas", model_choice="full")
base = base.non_dimensionalise()
base.far_dissolved_concentration_scaled = 0
results = solve(base)

gas_flow = PhysicalParams(name="saturated", model_choice="full")
gas_flow = gas_flow.non_dimensionalise()
results_gas = solve(gas_flow)

height = np.linspace(-2, 1, 1000)

fig = plt.figure(figsize=(4.5, 3.5), constrained_layout=True)
gs = fig.add_gridspec(ncols=3, nrows=1)
ax_left = plt.subplot(gs[0, 0])
ax_right = plt.subplot(gs[0, 1], sharey=ax_left)
ax_end = plt.subplot(gs[0, 2], sharey=ax_left)
ax_left.plot(results_gas.temperature(height), height, GREEN, label=results_gas.name)
ax_left.plot(results.temperature(height), height, "k", label=results.name)
ax_left.legend()
ax_right.plot(results_gas.solid_fraction(height) * 100, height, GREEN)
ax_right.plot(results.solid_fraction(height) * 100, height, "k")
ax_end.plot(results_gas.liquid_darcy_velocity(height), height, GREEN)
ax_end.plot(results.liquid_darcy_velocity(height), height, "k")
fig.suptitle(r"Simulation of steady mushy layer with no dissolved gas")
ax_left.set_xlabel(r"Non dimensional temperature $\theta$")
ax_right.set_xlabel(r"Solid fraction $\phi_s$ (\%)")
ax_end.set_xlabel(r"Liquid Darcy velocity $W_l$")
ax_left.set_ylabel(r"Scaled height $\eta$")
ax_left.set_xlim(-1.1, 0.1)
ax_left.set_ylim(-2, 1)
ax_right.set_xlim(-10, 110)
ax_end.set_xlim(-0.03, 0.005)
for ax in [ax_right, ax_left, ax_end]:
    ax.fill_between([-100, 110], -1, -3, color="#004488", alpha=0.2)
    ax.fill_between([-100, 110], -1, 0, color="#DDAA33", alpha=0.2)
    ax.fill_between([-100, 110], 0, 1, color="#BB5566", alpha=0.2)

plt.setp(ax_right.get_yticklabels(), visible=False)
plt.setp(ax_end.get_yticklabels(), visible=False)
plt.savefig("data/base_results.pdf")
plt.close()

"""fig 2
full model default behaviour for gas fraction and concentration."""
fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(ncols=2, nrows=1)
ax_left = plt.subplot(gs[0, 0])
ax_right = plt.subplot(gs[0, 1], sharey=ax_left)
ax_left.plot(results_gas.concentration(height), height, GREEN)
ax_right.plot(results_gas.gas_fraction(height) * 100, height, GREEN)
fig.suptitle(r"Simulation of steady mushy layer containing dissolved gas")
ax_left.set_xlabel(r"Dissolved gas concentration $\omega$")
ax_right.set_xlabel(r"Gas fraction $\phi_g$ (\%)")
ax_left.set_ylabel(r"Scaled height $\eta$")
ax_left.set_xlim(0.99, 1.1)
ax_left.set_ylim(-2, 1)
ax_right.set_xlim(-0.1, 3)
for ax in [ax_right, ax_left]:
    ax.fill_between([-100, 110], -1, -3, color="#004488", alpha=0.2)
    ax.fill_between([-100, 110], -1, 0, color="#DDAA33", alpha=0.2)
    ax.fill_between([-100, 110], 0, 1, color="#BB5566", alpha=0.2)

plt.setp(ax_right.get_yticklabels(), visible=False)
plt.savefig("data/gas_results.pdf")
plt.close()

"""fig 3
comparison of the different models for
solid fraction, temperature, gas fraction and liquid velocity
for default parameters so no gas buoyant rise."""
full = PhysicalParams(name="full", model_choice="full")
incompressible = PhysicalParams(name="incompressible", model_choice="incompressible")
thermally_ideal = PhysicalParams(name="thermally ideal", model_choice="ideal")
reduced = PhysicalParams(name="reduced", model_choice="reduced")
instant_nucleation = PhysicalParams(name="instant nucleation", model_choice="instant")

list_of_results = []
for parameters in [full, incompressible, thermally_ideal, reduced, instant_nucleation]:
    parameters = parameters.non_dimensionalise()
    list_of_results.append(solve(parameters))

fig = plt.figure(figsize=(5,5), constrained_layout=True)
gs = fig.add_gridspec(ncols=2, nrows=2)
ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[0, 1], sharey=ax1)
ax3 = plt.subplot(gs[1, 0])
ax4 = plt.subplot(gs[1, 1], sharey=ax3)

for results, color in zip(list_of_results, ["k", TEAL, ORANGE, RED, MAGENTA]):
    ax1.plot(results.temperature(height), height, color, label=results.name)
    ax2.plot(results.solid_fraction(height) * 100, height, color, label=results.name)
    ax3.plot(results.liquid_darcy_velocity(height), height, color, label=results.name)
    ax4.plot(results.gas_fraction(height) * 100, height, color, label=results.name)

fig.suptitle(r"Comparison of different models")
ax1.legend()
ax1.set_xlabel(r"Temperature $\theta$")
ax2.set_xlabel(r"Solid fraction $\phi_s$ (\%)")
ax3.set_xlabel(r"Liquid Darcy velocity $W_l$")
ax4.set_xlabel(r"Gas fraction $\phi_g$ (\%)")
ax1.set_ylabel(r"Scaled height $\eta$")
ax3.set_ylabel(r"Scaled height $\eta$")
ax1.set_xlim(-1.1, 0.1)
ax2.set_xlim(-10, 110)
ax3.set_xlim(-0.03, 0.005)
ax4.set_xlim(-0.1, 3)
ax1.set_ylim(-2, 1)
ax3.set_ylim(-2, 1)
for ax in [ax1, ax2, ax3, ax4]:
    ax.fill_between([-100, 110], -1, -3, color="#004488", alpha=0.2)
    ax.fill_between([-100, 110], -1, 0, color="#DDAA33", alpha=0.2)
    ax.fill_between([-100, 110], 0, 1, color="#BB5566", alpha=0.2)

plt.setp(ax2.get_yticklabels(), visible=False)
plt.setp(ax4.get_yticklabels(), visible=False)
plt.savefig("data/model_comparison.pdf")
plt.close()
