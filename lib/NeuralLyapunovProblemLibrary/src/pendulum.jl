@independent_variables t
Dt = Differential(t); DDt = Dt^2

@variables θ(t) τ(t) [input = true]
@parameters ζ ω_0

eqs = [DDt(θ) + 2ζ * ω_0 * Dt(θ) + ω_0^2 * sin(θ) ~ τ]

##################################### Driven pendulum ######################################
@named pendulum = ODESystem(
    eqs,
    t,
    [θ, τ],
    [ζ, ω_0]
)

eqs = [DDt(θ) + 2ζ * ω_0 * Dt(θ) + ω_0^2 * sin(θ) ~ 0]

#################################### Undriven pendulum #####################################
@named pendulum_undriven = ODESystem(
    eqs,
    t,
    [θ],
    [ζ, ω_0]
)

"""
    plot_pendulum(θ, t; title)
    plot_pendulum(sol; title, N, angle_symbol)

Plot the pendulum's trajectory.

# Arguments
  - `θ`: The angle of the pendulum at each time step.
  - `t`: The time steps.
  - `sol`: The solution to the ODE problem.

# Keyword arguments
  - `title`: The title of the plot; defaults to no title (i.e., `title=""`).
  - `N`: The number of points to plot; when using `θ` and `t`, uses `length(t)`; defaults to
    500 when using `sol`.
  - `angle_symbol`: The symbol of the angle in `sol`; defaults to `:θ`.
"""
function plot_pendulum end
