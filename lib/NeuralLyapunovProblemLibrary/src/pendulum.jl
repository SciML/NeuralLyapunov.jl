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

function plot_pendulum end
