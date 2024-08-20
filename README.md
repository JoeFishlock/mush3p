# mush3p #

## Installation ##
Run the command `pip install /path/to/this/directory/`.
This installs the necessary dependencies and allows mush3p to be imported.

## Usage ##

### Full Model ###
The steady ODE equations to be solved are
```math
\phi_s = 0 \qquad \text{and} \qquad \Theta_s \quad \text{arbitrary} \qquad \text{for} \qquad \theta > \theta_L,
```

```math
\theta = \theta_L(\Theta_l)\qquad \text{and} \qquad \Theta_s = - \mathcal{C}
\qquad \text{for} \qquad \theta_S \leq \theta \leq \theta_L,
```

```math
\phi_l = 0 \qquad \text{and} \qquad \Theta_l \quad \text{arbitrary} \qquad \text{for} \qquad \theta \leq \theta_S,
```

```math
\theta_L = - \Theta_l,
```

```math
\theta_S = -1,
```

```math
  \frac{\mathrm{d} \phi_g}{\mathrm{d} z} = \frac{\mathrm{d} W_l}{\mathrm{d} z},
```

```math
\frac{\mathrm{d}}{\mathrm{d}z}\left[ \phi_s \left( \Theta_s + \mathcal{C} \right) + \phi_l \left( \Theta_l+\mathcal{C} \right) \right] 
+ \frac{\mathrm{d} }{\mathrm{d} z}\left[ W_l\left( \Theta_l+\mathcal{C} \right)  \right] 
= 0,
```

```math
  \frac{\mathrm{d}}{\mathrm{d}z}\left( \omega W_l + \omega \phi_l \right)
  = -\text{Da} I(\omega, \, \sigma)\left( \omega-1 \right),
```

```math
  \frac{\mathrm{d}}{\mathrm{d}z}\left( \psi W_g + \psi \phi_g \right) 
  = \text{Da}\chi I(\omega, \, \sigma) (\omega-1),
```

```math
  \left( \phi_s + \phi_l \right) \frac{\mathrm{d}\theta}{\mathrm{d}z}
  + W_l \frac{\mathrm{d}\theta}{\mathrm{d}z}
  =  \text{St} \frac{\mathrm{d}\phi_s}{\mathrm{d}z}
  + \frac{\mathrm{d}}{\mathrm{d}z}
  \left( \left[ \phi_s + \phi_l + \nu_g \phi_g \right] \frac{\mathrm{d}\theta}{\mathrm{d}z} \right),
```

```math
  W_l = - \pi(\phi_l) \frac{\mathrm{d}p_H}{\mathrm{d}z}, \qquad \text{where} \qquad p_H = p_l - p_0 + p_0 \mathcal{H} z,
```

```math
\psi = \left( 1 + \frac{\theta}{\theta_K} \right)^{-1}
\left( 1 + \frac{p_H}{p_0} + \text{La} - \mathcal{H} z \right),
```

```math
  W_g = \phi_g \left( \frac{\mathcal{B} }{K_1(\lambda)}
  + \frac{2 G(\lambda)}{\phi_l} W_l \right),
  \qquad \text{where} \qquad \lambda = \frac{\lambda}{\phi_l^q},
```

```math
\pi(\phi_l) = \left( \frac{1}{\mathcal{K}} + \frac{(1-\phi_l)^2}{\phi_l^3} \right)^{-1}.
```

## License ##
[MIT](https://choosealicense.com/licenses/mit/)
