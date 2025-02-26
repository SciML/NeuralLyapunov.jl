using ModelingToolkit
import ModelingToolkit: inputs, generate_control_function
using NeuralLyapunovProblemLibrary
using OrdinaryDiffEq
using Plots
using LinearAlgebra
using ControlSystemsBase: lqr, Continuous
using Test

#################################### Hovering quadrotor ####################################
function π_vertical_only(x, p; y_goal=0.0, k_p=1.0, k_d=1.0)
    y, ẏ = x[2], x[5]
    m, I_quad, g, r = p
    T0 = m * g / 2
    T = T0 - k_p * m * g / r * (y - y_goal) - k_d * m * sqrt(g / r) * ẏ
    T = max(T, 0.0)
    return [T, T]
end

_, _, p, quadrotor_planar_simplified = generate_control_function(
    quadrotor_planar;
    simplify=true,
    split=false
)

t = independent_variable(quadrotor_planar)
Dt = Differential(t)
q = setdiff(unknowns(quadrotor_planar), inputs(quadrotor_planar))

params = map(Base.Fix1(getproperty, quadrotor_planar), toexpr.(p))
u = map(
        Base.Fix1(getproperty, quadrotor_planar),
        toexpr.(getproperty.(inputs(quadrotor_planar_simplified), :f))
)
q = map(
        Base.Fix1(getproperty, quadrotor_planar),
        toexpr.(getproperty.(q, :f))
)
x = vcat(q, Dt.(q))

@named vertical_only_controller = ODESystem(
    u .~ π_vertical_only(x, params),
    t,
    u,
    params
)

@named quadrotor_planar_vertical_only = compose(vertical_only_controller, quadrotor_planar)
quadrotor_planar_vertical_only = structural_simplify(quadrotor_planar_vertical_only)

# Hovering
# Assume rotors are negligible mass whne calculating the moment of inertia
x0 = Dict(x .=> zeros(6))
x0[q[2]] = rand()
x0[x[5]] = rand()
m, r = ones(2)
g = 1.0
I_quad = m * r^2 / 12
p = Dict(params .=> [m, I_quad, g, r])
τ = sqrt(r / g)

prob = ODEProblem(quadrotor_planar_vertical_only, x0, 15τ, p)
sol = solve(prob, Tsit5())

anim = plot_quadrotor_planar(
    sol,
    [m, I_quad, g, r];
    x_symbol=q[1],
    y_symbol=q[2],
    θ_symbol=q[3],
    u1_symbol=u[1],
    u2_symbol=u[2]
)
gif(anim, fps = 50)
