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
println("3D quadrotor vertical only test")

function π_vertical_only(x, p, t; z_goal = 0.0, k_p = 1.0, k_d = 1.0)
    z = x[3]
    ż = x[9]
    m, g = p[1:2]
    Izz = p[end]

    # Calculate a characteristic length scale for scaling the PD controller
    r = sqrt(Izz / m)
    k_p = k_p * m * g / r
    k_d = k_d * m * sqrt(g / r)

    # PD controller for vertical motion, limited to nonnegative thrust
    T0 = m * g
    T = T0 - k_p * (z - z_goal) - k_d * ż
    return [T, 0, 0, 0]
end

@named quadrotor_3d = Quadrotor3D()
@named quadrotor_3d_vertical_only = control_quadrotor_3d(quadrotor_3d, π_vertical_only)
quadrotor_3d_vertical_only = mtkcompile(quadrotor_3d_vertical_only)

# Hovering
# Assume rotors are negligible mass when calculating the moment of inertia
m, L = ones(2)
g = 1.0
Ixx = Iyy = m * L^2 / 6
Izz = m * L^2 / 3
Ixy = Ixz = Iyz = 0.0
p = [m, g, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
τ = sqrt(L / g)

q0 = [0, 0, rand(rng), 0, 0, 0]
q̇0 = [0, 0, rand(rng), 0, 0, rand(rng)]
x = get_quadrotor_3d_state_symbols(quadrotor_3d)
x0 = Dict(x .=> vcat(q0, q̇0))

p_dict = Dict(get_quadrotor_3d_param_symbols(quadrotor_3d) .=> p)
op = merge(x0, p_dict)
prob = ODEProblem(quadrotor_3d_vertical_only, op, 15τ)
sol = solve(prob, Tsit5())

q = x[1:6]
q̇ = x[7:12]
x_end, y_end, z_end, φ_end, θ_end, ψ_end = sol[q][end]
vx_end, vy_end, vz_end, ωφ_end, ωθ_end, ωψ_end = sol[q̇][end]
@test x_end ≈ 0.0 atol = 1.0e-4
@test y_end ≈ 0.0 atol = 1.0e-4
@test z_end ≈ 0.0 atol = 1.0e-4
@test φ_end ≈ 0.0 atol = 1.0e-4
@test θ_end ≈ 0.0 atol = 1.0e-4
@test vx_end ≈ 0.0 atol = 1.0e-4
@test vy_end ≈ 0.0 atol = 1.0e-4
@test vz_end ≈ 0.0 atol = 1.0e-4
@test ωφ_end ≈ 0.0 atol = 1.0e-4
@test ωθ_end ≈ 0.0 atol = 1.0e-4
@test ωψ_end ≈ q̇0[6] atol = 1.0e-4

u = get_quadrotor_3d_input_symbols(quadrotor_3d)
anim = plot_quadrotor_3d(
    sol,
    p;
    x_symbol = q[1],
    y_symbol = q[2],
    z_symbol = q[3],
    φ_symbol = q[4],
    θ_symbol = q[5],
    ψ_symbol = q[6],
    T_symbol = u[1],
    τφ_symbol = u[2],
    τθ_symbol = u[3],
    τψ_symbol = u[4]
)
@test anim isa Plots.Animation
# gif(anim, fps = 50)

############################## LQR planar quadrotor controller #############################
println("3D quadrotor LQR test")

function quadrotor_3d_lqr_matrix(
        p;
        x_eq = zeros(12),
        u_eq = [p[1] * p[2], 0, 0, 0],
        Q = I(12),
        R = I(4)
    )
    @named quad = Quadrotor3D()

    u = inputs(quad)
    x = setdiff(unknowns(quad), u)
    params = parameters(quad)

    op = Dict(vcat(x .=> x_eq, u .=> u_eq, params .=> p))

    mats, sys = linearize(quad, u, x; op)

    # Create permutation matrices Px : x_new = Px * x and Pu : u_new = Pu * u
    x_new = unknowns(sys)
    u_new = inputs(sys)

    Px = (x_new .- x') .=== 0
    Pu = (u_new .- u') .=== 0

    A = Px' * mats[:A] * Px
    B = Px' * mats[:B] * Pu

    return lqr(Continuous, A, B, Q, R)
end

function π_lqr(p; x_eq = zeros(12), u_eq = [p[1] * p[2], 0, 0, 0], Q = I(12), R = I(4))
    L = quadrotor_3d_lqr_matrix(p; Q, R, x_eq, u_eq)
    return (x, _p, _t) -> -L * (x - x_eq) + u_eq
end

@named quadrotor_3d = Quadrotor3D()

# Assume rotors are negligible mass when calculating the moment of inertia
m, L = ones(2)
g = 1.0
Ixx = Iyy = m * L^2 / 6
Izz = m * L^2 / 3
Ixy = Ixz = Iyz = 0.0
p = [m, g, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]

@named quadrotor_3d_lqr = control_quadrotor_3d(quadrotor_3d, π_lqr(p))
quadrotor_3d_lqr = mtkcompile(quadrotor_3d_lqr)

# Fly to origin
δ = 0.5
x = get_quadrotor_3d_state_symbols(quadrotor_3d)
x0 = Dict(x .=> δ .* (2 .* rand(rng, 12) .- 1))
p_dict = Dict(get_quadrotor_3d_param_symbols(quadrotor_3d) .=> p)
τ = sqrt(L / g)

op = merge(x0, p_dict)
prob = ODEProblem(quadrotor_3d_lqr, op, 15τ)
sol = solve(prob, Tsit5())

q = x[1:6]
q̇ = x[7:12]
x_end, y_end, z_end, φ_end, θ_end, ψ_end = sol[q][end]
vx_end, vy_end, vz_end, ωφ_end, ωθ_end, ωψ_end = sol[q̇][end]
@test x_end ≈ 0.0 atol = 1.0e-4
@test y_end ≈ 0.0 atol = 1.0e-4
@test z_end ≈ 0.0 atol = 1.0e-4
@test φ_end ≈ 0.0 atol = 1.0e-4
@test θ_end ≈ 0.0 atol = 1.0e-4
@test ψ_end ≈ 0.0 atol = 1.0e-4
@test vx_end ≈ 0.0 atol = 1.0e-4
@test vy_end ≈ 0.0 atol = 1.0e-4
@test vz_end ≈ 0.0 atol = 1.0e-4
@test ωφ_end ≈ 0.0 atol = 1.0e-4
@test ωθ_end ≈ 0.0 atol = 1.0e-4
@test ωψ_end ≈ 0.0 atol = 1.0e-4

u = get_quadrotor_3d_input_symbols(quadrotor_3d)
anim = plot_quadrotor_3d(
    sol,
    [m, g, Ixx, Ixy, Ixz, Iyy, Iyz, Izz];
    x_symbol = q[1],
    y_symbol = q[2],
    z_symbol = q[3],
    φ_symbol = q[4],
    θ_symbol = q[5],
    ψ_symbol = q[6],
    T_symbol = u[1],
    τφ_symbol = u[2],
    τθ_symbol = u[3],
    τψ_symbol = u[4]
)
@test anim isa Plots.Animation
# gif(anim, fps = 50)
