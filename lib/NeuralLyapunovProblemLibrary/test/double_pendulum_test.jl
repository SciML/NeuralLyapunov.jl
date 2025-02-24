using ModelingToolkit
import ModelingToolkit: inputs, generate_control_function
using NeuralLyapunovProblemLibrary
using OrdinaryDiffEq
using Plots
using LinearAlgebra
using Test

############### Undriven double pendulum should drop to downward equilibrium ###############
x0 = vcat(2π * rand(2) .- π, rand(2))

# Assume uniform rods of random mass and length
m1, m2 = rand(2)
l1, l2 = rand(2)
lc1, lc2 = l1 /2, l2 / 2
I1 = m1 * l1^2 / 3
I2 = m2 * l2^2 / 3
g = 1.0
p = [I1, I2, l1, l2, lc1, lc2, m1, m2, g]

prob = ODEProblem(structural_simplify(double_pendulum_undriven), x0, 100, p)
sol = solve(prob, Tsit5())

anim = plot_double_pendulum(sol, p)
@test anim isa Plots.Animation
# gif(anim, fps=50)

########################### Feedback cancellation, PD controller ###########################
function π_cancellation(x, p)
    θ1, θ2, ω1, ω2 = x
    I1, I2, l1, _, lc1, lc2, m1, m2, g = p
    M = [
        I1 + I2 + m2 * l1^2 + 2 * m2 * l1 * lc2 * cos(θ2)   I2 + m2 * l1 * lc2 * cos(θ2);
        I2 + m2 * l1 * lc2 * cos(θ2)                        I2
    ]
    G = [
        -m1 * g * lc1 * sin(θ1) - m2 * g * (l1 * sin(θ1) + lc2 * sin(θ1 + θ2));
        -m2 * g * lc2 * sin(θ1 + θ2)
    ]
    return -0.1 * M \ ([θ1, θ2] .- [π, π] + [ω1, ω2]) - G
end

_, x, p, double_pendulum_simplified = generate_control_function(
    double_pendulum;
    simplify=true,
    split=false
)

t = independent_variable(double_pendulum)
Dt = Differential(t)

p = map(Base.Fix1(getproperty, double_pendulum), toexpr.(p))
u = map(
        Base.Fix1(getproperty, double_pendulum),
        toexpr.(getproperty.(inputs(double_pendulum_simplified), :f))
)
x = [double_pendulum.θ1, double_pendulum.θ2, Dt(double_pendulum.θ1), Dt(double_pendulum.θ2)]

@named cancellation_controller = ODESystem(
    u .~ π_cancellation(x, p),
    t,
    u,
    p
)
@named double_pendulum_feedback_cancellation = compose(cancellation_controller, double_pendulum)
double_pendulum_feedback_cancellation = structural_simplify(double_pendulum_feedback_cancellation)

# Swing up to upward equilibrium
# Assume uniform rods of random mass and length
m1, m2 = rand(2)
l1, l2 = rand(2)
lc1, lc2 = l1 /2, l2 / 2
I1 = m1 * l1^2 / 3
I2 = m2 * l2^2 / 3
g = 1.0
p = [I1, I2, l1, l2, lc1, lc2, m1, m2, g]

x0 = vcat(2π * rand(2) .- π, rand(2))

prob = ODEProblem(double_pendulum_feedback_cancellation, x0, 100, p)
sol = solve(prob, Tsit5())
θ1_end, ω1_end = sol[:double_pendulum₊θ1][end], sol.u[end][3]
x1_end, y1_end = sin(θ1_end), -cos(θ1_end)
θ2_end, ω2_end = sol[:double_pendulum₊θ2][end], sol.u[end][4]
x2_end, y2_end = sin(θ2_end), -cos(θ2_end)
@test sqrt(sum(abs2, [x1_end, y1_end] .- [0, 1])) ≈ 0 atol=1e-4
@test sqrt(sum(abs2, [x2_end, y2_end] .- [0, 1])) ≈ 0 atol=1e-4
@test ω1_end ≈ 0 atol=1e-4
@test ω2_end ≈ 0 atol=1e-4

#=
gif(
    plot_double_pendulum(
        sol,
        p;
        angle1_symbol=:double_pendulum₊θ1,
        angle2_symbol=:double_pendulum₊θ2
    ),
    fps=50
)
=#
