using ModelingToolkit
import ModelingToolkit: inputs, generate_control_function
using NeuralLyapunovProblemLibrary
using OrdinaryDiffEq
using Plots
using LinearAlgebra
using ControlSystemsBase: lqr, Continuous
using Test

#################################### Hovering quadrotor ####################################
function π_vertical_only(x, p; z_goal=0.0, k_p=1.0, k_d=1.0)
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
    T = max(T, 0.0)
    return [T, 0, 0, 0]
end

_, _, p, quadrotor_3d_simplified = generate_control_function(
    quadrotor_3d;
    simplify=true,
    split=false
)

t = independent_variable(quadrotor_3d)
Dt = Differential(t)
q = setdiff(unknowns(quadrotor_3d), inputs(quadrotor_3d))[1:6]
q̇ = vcat(Dt.(q[1:3]), setdiff(unknowns(quadrotor_3d), inputs(quadrotor_3d))[7:9])
x = vcat(q, q̇)

params = map(Base.Fix1(getproperty, quadrotor_3d), toexpr.(p))
u = map(
        Base.Fix1(getproperty, quadrotor_3d),
        toexpr.(getproperty.(inputs(quadrotor_3d_simplified), :f))
)
q = map(
        Base.Fix1(getproperty, quadrotor_3d),
        toexpr.(getproperty.(q, :f))
)
q̇ = vcat(
        Dt.(q[1:3]),
        map(
            Base.Fix1(getproperty, quadrotor_3d),
            toexpr.(getproperty.(q̇[4:6], :f))
        )
)
x = vcat(q, q̇)

@named vertical_only_controller = ODESystem(
    u .~ π_vertical_only(x, params),
    t,
    vcat(x, u),
    params
)

@named quadrotor_3d_vertical_only = compose(vertical_only_controller, quadrotor_3d)
quadrotor_3d_vertical_only = structural_simplify(quadrotor_3d_vertical_only)

# Hovering
# Assume rotors are negligible mass when calculating the moment of inertia
m, L = ones(2)
g = 1.0
Ixx = Iyy = m * L^2 / 6
Izz = m * L^2 / 3
Ixy = Ixz = Iyz = 0.0
p = Dict(params .=> [m, g, Ixx, Ixy, Ixz, Iyy, Iyz, Izz])

τ = sqrt(L / g)

x0 = Dict(x .=> zeros(12))
x0[q[3]] = rand()
x0[q̇[3]] = rand()
x0[q̇[6]] = rand()

prob = ODEProblem(quadrotor_3d_vertical_only, x0, 15τ, p)
sol = solve(prob, Tsit5())

x_end, y_end, z_end = sol[q][end]
v_x_end, v_y_end, v_θ_end = sol[Dt.(q)][end]
@test x_end ≈ 0.0 atol=1e-4
@test y_end ≈ 0.0 atol=1e-4
@test θ_end ≈ 0.0 atol=1e-4
@test v_x_end ≈ 0.0 atol=1e-4
@test v_y_end ≈ 0.0 atol=1e-4
@test v_θ_end ≈ 0.0 atol=1e-4

anim = plot_quadrotor_3d(
    sol,
    [m, g, Ixx, Ixy, Ixz, Iyy, Iyz, Izz];
    x_symbol=q[1],
    y_symbol=q[2],
    z_symbol=q[3],
    φ_symbol=q[4],
    θ_symbol=q[5],
    ψ_symbol=q[6],
    T_symbol=u[1],
    τφ_symbol=u[2],
    τθ_symbol=u[3],
    τψ_symbol=u[4]
)
@test anim isa Plots.Animation
gif(anim, fps = 50)
