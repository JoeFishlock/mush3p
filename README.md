# mush3p #

Mush3p is a Python package which simulate a steadily solidifying three-phase mushy layer growing from a liquid melt containing dissolved gas.
The liquid, which contains dissolved solute and gas, is translated upwards at a constant velocity towards a cold heat exhanger at the eutectic temperature of the system.
The problem set up and underlying conservation equations are detailed in an upcoming paper [To be added here].
Here is an example of a simulation of the gas fraction distribution in a steady mushy layer at different values of the concentration ratio dimensionless parameter.


![Simulation of three-phase mushy layers at different concentration ratios from an upcoming paper](example_plot.svg)

## Installation ##
Mush3p is available on PyPI and can be installed with pip via
```bash
pip install mush3p
```

## Usage ##
Mush3p provides the classes `PhysicalParams` and `NonDimensionalParams` to define a simulation configuration using dimensional or non-dimensional parameters respectively.
The following python code illustrates how to set up and solve a simulation.
```python
import mush3p as m3p

parameters = m3p.PhysicalParams(name="example", model_choice="full", bubble_radius=1e-3)
non_dimensional_parameters = parameters.non_dimensionalise()
solution = m3p.solve(non_dimensional_parameters)
```

Mush3p returns the solution as a `NonDimensionalResults` object which contains methods to calculate the solution profiles of quantities of interest as functions of height scaled by the mushy layer depth.
These methods interpolate the solution outside of the mushy layer using known analytical solutions.
The steady mushy layer exists between scaled heights of -1 and 0, with eutectic solid above and liquid below.
The following python code illustrates how to plot some solution profiles.
```python
import matplotlib.pyplot as plt
import numpy as np

# vertical coordinate to plot solution for
height = np.linspace(-1.2, 0.2, 100)

# plot four of the solution profiles side by side
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, sharey=True)

ax1.plot(solution.temperature(height), height)
ax1.set_ylabel("depth")
ax1.set_title("temperature")

ax2.plot(solution.solid_fraction(height), height)
ax2.set_title("solid fraction")

ax3.plot(solution.gas_fraction(height), height)
ax3.set_title("gas fraction")

ax4.plot(solution.concentration(height), height)
ax4.set_title("dissolved gas concentration")

plt.show()
```

## Available models ##
Details of the available models can be found in the accompanying upcoming paper [To be added here].
In brief there are four options for the `model_choice` parameter:
- `"full"`: the full model with no approximations.
- `"incompressible"`: the full model but with an incompressible gas phase.
- `"reduced"`: the reduced approximation scheme where the volume of gas exsolved is small and incompressible and drives no liquid flow within the mushy layer. Under this approximation scheme the gas volume fraction is neglected and so solid and liquid volume fractions sum to one.
- `"instant"`: the reduced model with instantaneous exsolution of any dissolved gas supersaturation into the gas phase. This corresponds to the limit where the Damkohler number goes to infinity.

## Tests ##
The tests directory contains tests of simulations for a variety of input parameters.
Run the tests using `pytest` from the root directory of the repository.

## Docs ##
API reference documentation built using `mkdocs gh-deploy` is available at
[documentation](https://JoeFishlock.github.io/mush3p).

## License ##
[MIT](https://choosealicense.com/licenses/mit/)
