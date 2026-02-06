using ModelingToolkit
import ModelingToolkit: inputs, D_nounits as Dt
using NeuralLyapunovProblemLibrary
using OrdinaryDiffEq
using Plots
using LinearAlgebra
using ControlSystemsBase: lqr, Continuous
using Test, StableRNGs

rng = StableRNG(0)

################################## Double pendulum energy ##################################
function U(x, p)
    θ1, θ2, _, _ = x
    I1, I2, l1, l2, lc1, lc2, m1, m2, g = p
    return -m1 * g * lc1 * cos(θ1) - m2 * g * (l1 * cos(θ1) + lc2 * cos(θ1 + θ2))
end
function T(x, p)
    θ1, θ2, ω1, ω2 = x
    I1, I2, l1, l2, lc1, lc2, m1, m2, g = p
    M = [
        I1 + I2 + m2 * l1^2 + 2 * m2 * l1 * lc2 * cos(θ2) I2 + m2 * l1 * lc2 * cos(θ2);
        I2 + m2 * l1 * lc2 * cos(θ2) I2
    ]
    return 0.5 * dot([ω1, ω2], M, [ω1, ω2])
end
E(x, p) = T(x, p) + U(x, p)

######################### Undriven double pendulum conserve energy #########################
println("Undriven double pendulum energy conservation test")

@mtkcompile double_pendulum_undriven = DoublePendulum(; actuation = :undriven)

# Assume uniform rods of random mass and length
m1, m2 = ones(2)
l1, l2 = ones(2)
lc1, lc2 = l1 / 2, l2 / 2
I1 = m1 * l1^2 / 3
I2 = m2 * l2^2 / 3
g = 1.0
p = [I1, I2, l1, l2, lc1, lc2, m1, m2, g]

x = get_double_pendulum_state_symbols(double_pendulum_undriven)
x0 = Dict(x .=> vcat(2π * rand(rng, 2) .- π, zeros(2)))

params = get_double_pendulum_param_symbols(double_pendulum_undriven)
p_dict = Dict(params .=> p)

op = merge(x0, p_dict)
prob = ODEProblem(double_pendulum_undriven, op, 100)
sol = solve(prob, Tsit5(), abstol = 1.0e-10, reltol = 1.0e-10)

# Test energy conservation
samples = sol[x]

potential_energy = map(Base.Fix2(U, p), samples)
kinetic_energy = map(Base.Fix2(T, p), samples)
total_energy = map(Base.Fix2(E, p), samples)

#=
plot(
    sol.t,
    [potential_energy, kinetic_energy, total_energy],
    labels=["Potential energy" "Kinetic Energy" "Total Energy"],
    xlabel="Time",
    ylabel="Energy"
)
=#

avg_energy = sum(total_energy) / length(total_energy)
@test maximum(abs, total_energy .- avg_energy) / abs(avg_energy) < 1.0e-4

# Test plotting extension
anim = plot_double_pendulum(sol, p)
@test anim isa Plots.Animation
# gif(anim, fps=50)

########################### Feedback cancellation, PD controller ###########################
println("Double pendulum feedback cancellation test")

# Assume uniform rods of random mass and length
m1, m2 = ones(2)
l1, l2 = ones(2)
lc1, lc2 = l1 / 2, l2 / 2
I1 = m1 * l1^2 / 3
I2 = m2 * l2^2 / 3
g = 1.0
p = [I1, I2, l1, l2, lc1, lc2, m1, m2, g]

@named double_pendulum = DoublePendulum(param_defaults = p)

function π_cancellation(x, p, t)
    θ1, θ2, ω1, ω2 = x
    I1, I2, l1, l2, lc1, lc2, m1, m2, g = p
    M = [
        I1 + I2 + m2 * l1^2 + 2 * m2 * l1 * lc2 * cos(θ2) I2 + m2 * l1 * lc2 * cos(θ2);
        I2 + m2 * l1 * lc2 * cos(θ2) I2
    ]
    G = [
        -m1 * g * lc1 * sin(θ1) - m2 * g * (l1 * sin(θ1) + lc2 * sin(θ1 + θ2));
        -m2 * g * lc2 * sin(θ1 + θ2)
    ]
    return -0.1 * M \ ([θ1, θ2] .- [π, π] + [ω1, ω2]) - G
end

@mtkcompile double_pendulum_feedback_cancellation = control_double_pendulum(double_pendulum, π_cancellation)

# Swing up to upward equilibrium
x = get_double_pendulum_state_symbols(double_pendulum)
x0 = Dict(x .=> vcat(2π * rand(rng, 2) .- π, rand(rng, 2)))

prob = ODEProblem(double_pendulum_feedback_cancellation, x0, 100)
sol = solve(prob, Tsit5())

θ1 = double_pendulum.θ1
θ2 = double_pendulum.θ2

θ1_end, ω1_end = sol[θ1][end], sol[Dt(θ1)][end]
x1_end, y1_end = sin(θ1_end), -cos(θ1_end)
θ2_end, ω2_end = sol[θ2][end], sol[Dt(θ2)][end]
x2_end, y2_end = sin(θ2_end), -cos(θ2_end)
@test sqrt(sum(abs2, [x1_end, y1_end] .- [0, 1])) ≈ 0 atol = 1.0e-4
@test sqrt(sum(abs2, [x2_end, y2_end] .- [0, 1])) ≈ 0 atol = 1.0e-4
@test ω1_end ≈ 0 atol = 1.0e-4
@test ω2_end ≈ 0 atol = 1.0e-4

# gif(plot_double_pendulum(sol, p; angle1_symbol=θ1, angle2_symbol=θ2), fps=50)

################################## LQR acrobot controller ##################################
println("Acrobot LQR test")

function acrobot_lqr_matrix(p; x_eq = [π, 0, 0, 0], Q = I(4), R = I(1))
    I1, I2, l1, l2, lc1, lc2, m1, m2, g = p
    θ1, θ2 = x_eq[1:2]

    # Assumes linearization around a fixed point
    M = [
        I1 + I2 + m2 * l1^2 + 2 * m2 * l1 * lc2 * cos(θ2) I2 + m2 * l1 * lc2 * cos(θ2);
        I2 + m2 * l1 * lc2 * cos(θ2) I2
    ]
    B = [0, 1]
    Jτ_g = [
        -m1 * g * lc1 * cos(θ1) - m2 * g * (l1 * cos(θ1) + lc2 * cos(θ1 + θ2)) -m2 * g * lc2 * cos(θ1 + θ2);
        -m2 * g * lc2 * cos(θ1 + θ2) -m2 * g * lc2 * cos(θ1 + θ2)
    ]

    A_lin = [
        zeros(2, 2) I(2);
        M \ Jτ_g zeros(2, 2)
    ]
    B_lin = [zeros(2); M \ B]

    return lqr(Continuous, A_lin, B_lin, Q, R)
end

function π_lqr(p; x_eq = [π, 0, 0, 0], Q = I(4), R = I(1))
    L = acrobot_lqr_matrix(p; x_eq, Q, R)
    return (x, _p, _t) -> -L * (x .- x_eq)
end

@named acrobot = Acrobot()

# Assume uniform rods of random mass and length
m1, m2 = ones(2)
l1, l2 = ones(2)
lc1, lc2 = l1 / 2, l2 / 2
I1 = m1 * l1^2 / 3
I2 = m2 * l2^2 / 3
g = 1.0
p = [I1, I2, l1, l2, lc1, lc2, m1, m2, g]

@mtkcompile acrobot_lqr = control_double_pendulum(acrobot, π_lqr(p; x_eq = [π, π, 0, 0]))

# Remain close to upward equilibrium
x = get_double_pendulum_state_symbols(acrobot)
x0 = Dict(x .=> [π, π, 0, 0] + 0.005 * vcat(2π * rand(rng, 2) .- π, 2 * rand(rng, 2) .- 1))
p_dict = Dict(get_double_pendulum_param_symbols(acrobot) .=> p)
tspan = 1000

op = merge(x0, p_dict)
prob = ODEProblem(acrobot_lqr, op, tspan)
sol = solve(prob, Tsit5())

θ1, θ2, ω1, ω2 = x

anim = plot_double_pendulum(sol, p; angle1_symbol = θ1, angle2_symbol = θ2)
@test anim isa Plots.Animation
# gif(anim, fps=50)

θ1_end, ω1_end = sol[θ1][end], sol[ω1][end]
x1_end, y1_end = sin(θ1_end), -cos(θ1_end)
θ2_end, ω2_end = sol[θ2][end], sol[ω2][end]
x2_end, y2_end = sin(θ2_end), -cos(θ2_end)
@test sqrt(sum(abs2, [x1_end, y1_end] .- [0, 1])) ≈ 0 atol = 1.0e-4
@test sqrt(sum(abs2, [x2_end, y2_end] .- [0, 1])) ≈ 0 atol = 1.0e-4
@test ω1_end ≈ 0 atol = 1.0e-4
@test ω2_end ≈ 0 atol = 1.0e-4
