using ModelingToolkit
import ModelingToolkit: inputs
using NeuralLyapunovProblemLibrary
using OrdinaryDiffEq
using Plots
using LinearAlgebra
using ControlSystemsBase: lqr, Continuous
using Test, StableRNGs

rng = StableRNG(0)

#################################### Hovering quadrotor ####################################
println("Planar quadrotor vertical only test")

@named quadrotor_planar = QuadrotorPlanar()

function π_vertical_only(x, p, t; y_goal = 0.0, k_p = 1.0, k_d = 1.0)
    y, ẏ = x[2], x[5]
    m, I_quad, g, r = p
    T0 = m * g / 2
    T = T0 - k_p * m * g / r * (y - y_goal) - k_d * m * sqrt(g / r) * ẏ
    return [T, T]
end

@named quadrotor_planar_vertical_only = control_quadrotor_planar(
    quadrotor_planar,
    π_vertical_only
)
quadrotor_planar_vertical_only = structural_simplify(quadrotor_planar_vertical_only)

# Hovering
# Assume rotors are negligible mass when calculating the moment of inertia
m, r = ones(2)
g = 1.0
I_quad = m * r^2 / 12
p = [m, I_quad, g, r]
τ = sqrt(r / g)

x = get_quadrotor_planar_state_symbols(quadrotor_planar)
x0 = Dict(x .=> [0, rand(rng), 0, 0, rand(rng), 0])
p_dict = Dict(get_quadrotor_planar_param_symbols(quadrotor_planar) .=> p)

prob = ODEProblem(quadrotor_planar_vertical_only, x0, 15τ, p_dict)
sol = solve(prob, Tsit5())

q = x[1:3]
q̇ = x[4:6]
x_end, y_end, θ_end = sol[q][end]
v_x_end, v_y_end, v_θ_end = sol[q̇][end]
@test x_end ≈ 0.0 atol = 1.0e-4
@test y_end ≈ 0.0 atol = 1.0e-4
@test θ_end ≈ 0.0 atol = 1.0e-4
@test v_x_end ≈ 0.0 atol = 1.0e-4
@test v_y_end ≈ 0.0 atol = 1.0e-4
@test v_θ_end ≈ 0.0 atol = 1.0e-4

u1, u2 = get_quadrotor_planar_input_symbols(quadrotor_planar)
anim = plot_quadrotor_planar(
    sol,
    p;
    x_symbol = q[1],
    y_symbol = q[2],
    θ_symbol = q[3],
    u1_symbol = u1,
    u2_symbol = u2
)
@test anim isa Plots.Animation
# gif(anim, fps = 50)

############################## LQR planar quadrotor controller #############################
println("Planar quadrotor LQR test")

function quadrotor_planar_lqr_matrix(p; Q = I(6), R = I(2))
    m, I_quad, g, r = p

    # Assumes linearization around a fixed point
    # x_eq = (x*, y*, 0, 0, 0, 0), u_eq = (mg / 2, mg / 2)
    A_lin = zeros(6, 6)
    A_lin[1:3, 4:6] .= I(3)
    A_lin[4, 3] = -g

    B_lin = zeros(6, 2)
    B_lin[5, :] .= 1 / m
    B_lin[6, :] .= r / I_quad, -r / I_quad

    return lqr(Continuous, A_lin, B_lin, Q, R)
end

function π_lqr(p; x_eq = zeros(6), Q = I(6), R = I(2))
    L = quadrotor_planar_lqr_matrix(p; Q, R)
    m, _, g, _ = p
    T0 = m * g / 2
    return (x, _p, _t) -> -L * (x - x_eq) + [T0, T0]
end

@named quadrotor_planar = QuadrotorPlanar()

# Assume rotors are negligible mass when calculating the moment of inertia
m, r = ones(2)
g = 1.0
I_quad = m * r^2 / 12
p = [m, I_quad, g, r]

@named quadrotor_planar_lqr = control_quadrotor_planar(quadrotor_planar, π_lqr(p))
quadrotor_planar_lqr = structural_simplify(quadrotor_planar_lqr)

# Fly to origin
x = get_quadrotor_planar_state_symbols(quadrotor_planar)
x0 = Dict(x .=> 2 * rand(rng, 6) .- 1)
p_dict = Dict(get_quadrotor_planar_param_symbols(quadrotor_planar) .=> p)
τ = sqrt(r / g)

prob = ODEProblem(quadrotor_planar_lqr, x0, 15τ, p_dict)
sol = solve(prob, Tsit5())

q = x[1:3]
q̇ = x[4:6]
x_end, y_end, θ_end = sol[q][end]
v_x_end, v_y_end, v_θ_end = sol[q̇][end]
@test x_end ≈ 0.0 atol = 1.0e-4
@test y_end ≈ 0.0 atol = 1.0e-4
@test θ_end ≈ 0.0 atol = 1.0e-4
@test v_x_end ≈ 0.0 atol = 1.0e-4
@test v_y_end ≈ 0.0 atol = 1.0e-4
@test v_θ_end ≈ 0.0 atol = 1.0e-4

u1, u2 = get_quadrotor_planar_input_symbols(quadrotor_planar)
anim = plot_quadrotor_planar(
    sol,
    p;
    x_symbol = q[1],
    y_symbol = q[2],
    θ_symbol = q[3],
    u1_symbol = u1,
    u2_symbol = u2
)
@test anim isa Plots.Animation
# gif(anim, fps = 50)
