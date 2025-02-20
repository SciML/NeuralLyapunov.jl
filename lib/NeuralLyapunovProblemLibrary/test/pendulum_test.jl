using ModelingToolkit
import ModelingToolkit: inputs, generate_control_function
using NeuralLyapunovProblemLibrary
using OrdinaryDiffEq
using Test

################## Undriven pendulum should drop to downward equilibrium ###################
x0 = π * rand(2)
p = rand(2)
τ = 1 / prod(p)
prob = ODEProblem(structural_simplify(pendulum_undriven), x0, 15τ, p)
sol = solve(prob, Tsit5())
x_end, y_end, ω_end = sin(sol.u[end][1]), -cos(sol.u[end][1]), sol.u[end][2]
@test sqrt(sum(abs2, [x_end, y_end] .- [0, -1])) ≈ 0 atol=1e-4
@test ω_end ≈ 0 atol=1e-4

using Plots

@test plot_pendulum(sol) isa Plots.Animation

############################# Feedback cancellation controller #############################
π_cancellation(x, p) = 2 * p[2]^2 * sin(x[1])

_, x, p, pendulum_simplified = generate_control_function(pendulum; simplify=true, split=false)

t = independent_variable(pendulum)
Dt = Differential(t)

p = map(Base.Fix1(getproperty, pendulum), toexpr.(p))
u = map(
        Base.Fix1(getproperty, pendulum),
        toexpr.(getproperty.(inputs(pendulum_simplified), :f))
)

@named cancellation_controller = ODESystem(
    u .~ π_cancellation([pendulum.θ, Dt(pendulum.θ)], p),
    t,
    u,
    p
)
@named pendulum_feedback_cancellation = compose(cancellation_controller, pendulum)
pendulum_feedback_cancellation = structural_simplify(pendulum_feedback_cancellation)

# Swing up to upward equilibrium
x0 = rand(2)
p = rand(2)
τ = 1 / prod(p)
prob = ODEProblem(pendulum_feedback_cancellation, x0, 15τ, p)
sol = solve(prob, Tsit5()); plot(sol)
x_end, y_end, ω_end = sin(sol.u[end][1]), -cos(sol.u[end][1]), sol.u[end][2]
@test sqrt(sum(abs2, [x_end, y_end] .- [0, 1])) ≈ 0 atol=1e-4
@test ω_end ≈ 0 atol=1e-4
