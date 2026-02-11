using ModelingToolkit
using ModelingToolkit: D_nounits as Dt
using NeuralLyapunovProblemLibrary
using OrdinaryDiffEq
using Plots
using Test, StableRNGs

rng = StableRNG(0)

################## Undriven pendulum should drop to downward equilibrium ###################
println("Undriven pendulum test")

@mtkcompile pendulum_undriven = Pendulum(; driven = false)

x0 = π * rand(rng, 2)
p = rand(rng, 2)
τ = 1 / prod(p)

op = Dict(
    vcat(
        get_pendulum_state_symbols(pendulum_undriven),
        get_pendulum_param_symbols(pendulum_undriven)
    ) .=> vcat(x0, p)
)
prob = ODEProblem(pendulum_undriven, op, 15τ)
sol = solve(prob, Tsit5())

θ = pendulum_undriven.θ
x_end, y_end, ω_end = sin(sol[θ][end]), -cos(sol[θ][end]), sol[Dt(θ)][end]
@test sqrt(sum(abs2, [x_end, y_end] .- [0, -1])) ≈ 0 atol = 1.0e-4
@test ω_end ≈ 0 atol = 1.0e-4

anim = plot_pendulum(sol)
@test anim isa Plots.Animation
# gif(anim, fps=50)

############################# Feedback cancellation controller #############################
println("Simple pendulum feedback cancellation test")

x0 = rand(rng, 2)
p = rand(rng, 2)
τ = 1 / prod(p)

@named pendulum_driven = Pendulum(defaults = p)

π_cancellation(x, p, t) = 2 * p[2]^2 * sin(x[1])

@mtkcompile pendulum_feedback_cancellation = control_pendulum(
    pendulum_driven,
    π_cancellation
)

# Swing up to upward equilibrium
op = Dict(get_pendulum_state_symbols(pendulum_driven) .=> x0)

prob = ODEProblem(pendulum_feedback_cancellation, op, 15τ)
sol = solve(prob, Tsit5())

θ = pendulum_driven.θ
x_end, y_end, ω_end = sin(sol[θ][end]), -cos(sol[θ][end]), sol[Dt(θ)][end]
@test sqrt(sum(abs2, [x_end, y_end] .- [0, 1])) ≈ 0 atol = 1.0e-4
@test ω_end ≈ 0 atol = 1.0e-4

anim = plot_pendulum(sol; angle_symbol = θ)
@test anim isa Plots.Animation
# gif(anim, fps=50)
