@parameters ζ ω_0

@independent_variables t
@variables θ(t) τ(t) [input = true]
Dt = Differential(t)
DDt = Dt^2

eqs = [DDt(θ) + 2ζ * ω_0 * Dt(θ) + ω_0^2 * sin(θ) ~ τ]

@named pendulum = ODESystem(
    eqs,
    t,
    [θ, τ],
    [ζ, ω_0]
)

eqs = [DDt(θ) + 2ζ * ω_0 * Dt(θ) + ω_0^2 * sin(θ) ~ 0]

@named pendulum_undriven = ODESystem(
    eqs,
    t,
    [θ],
    [ζ, ω_0]
)
