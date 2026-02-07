# Quadrotor Models

Two versions of the quadrotor are provided: a planar approximation ([`QuadrotorPlanar`](@ref)) and a 3D model ([`Quadrotor3D`](@ref)).
Additionally, when also using the Plots.jl package, the convenience plotting functions [`plot_quadrotor_planar`](@ref) and [`plot_quadrotor_3d`](@ref) are provided.

```@contents
Pages = ["quadrotor.md"]
Depth = 2:3
```

## Planar Approximation

The planar quadrotor ([`QuadrotorPlanar`](@ref), technically a birotor) is a rigid body with two rotors in line with the center of mass.

![Planar quadrotor animation](../imgs/quadrotor_planar.gif)

```@docs
QuadrotorPlanar
control_quadrotor_planar
get_quadrotor_planar_state_symbols
get_quadrotor_planar_param_symbols
get_quadrotor_planar_input_symbols
```

### Copy-Pastable Code

```@example plot_quadrotor_planar
using Random; Random.seed!(200) # hide
using ModelingToolkit
using NeuralLyapunovProblemLibrary
using OrdinaryDiffEq
using Plots
using LinearAlgebra
using ControlSystemsBase: lqr, Continuous

function π_lqr(p; x_eq = zeros(6), Q = I(6), R = I(2))
    m, I_quad, g, r = p

    # Assumes linearization around a fixed point
    # x_eq = (x*, y*, 0, 0, 0, 0), u_eq = (mg / 2, mg / 2)
    A_lin = zeros(6, 6)
    A_lin[1:3,4:6] .= I(3)
    A_lin[4,3] = -g

    B_lin = zeros(6, 2)
    B_lin[5,:] .= 1 / m
    B_lin[6,:] .= r / I_quad, -r / I_quad

    K = lqr(Continuous, A_lin, B_lin, Q, R)

    T0 = m * g / 2
    return (x, _p, _t) -> -K * (x - x_eq) + [T0, T0]
end

@named quadrotor_planar = QuadrotorPlanar()

# Assume rotors are negligible mass when calculating the moment of inertia
m, r = ones(2)
g = 1.0
I_quad = m * r^2 / 12
p = [m, I_quad, g, r]

@mtkcompile quadrotor_planar_lqr = control_quadrotor_planar(quadrotor_planar, π_lqr(p))

# Random initialization
x = get_quadrotor_planar_state_symbols(quadrotor_planar)
x0 = Dict(x .=> 2 * rand(6) .- 1)
params = get_quadrotor_planar_param_symbols(quadrotor_planar)
p_dict = Dict(params .=> p)
op = merge(x0, p_dict)
τ = sqrt(r / g)

prob = ODEProblem(quadrotor_planar_lqr, op, 15τ)
sol = solve(prob, Tsit5())

u = get_quadrotor_planar_input_symbols(quadrotor_planar)
gif(
    plot_quadrotor_planar(
        sol,
        p;
        x_symbol=x[1],
        y_symbol=x[2],
        θ_symbol=x[3],
        u1_symbol=u[1],
        u2_symbol=u[2]
    );
    fps = 50
)
```

### Plotting the Planar Quadrotor

```@docs
plot_quadrotor_planar
```

## 3D Model

A full 3D model from [quadrotor](@cite) is provided via [`Quadrotor3D`](@ref).

![3D quadrotor animation](../imgs/quadrotor_3d.gif)

```@docs
Quadrotor3D
control_quadrotor_3d
get_quadrotor_3d_state_symbols
get_quadrotor_3d_param_symbols
get_quadrotor_3d_input_symbols
```

### Copy-Pastable Code

```@example plot_quadrotor_3d
using Random; Random.seed!(200) # hide
using ModelingToolkit
import ModelingToolkit: inputs
using NeuralLyapunovProblemLibrary
using OrdinaryDiffEq
using Plots
using LinearAlgebra
using ControlSystemsBase: lqr, Continuous

# Define LQR controller
function π_lqr(p; x_eq = zeros(12), u_eq = [p[1]*p[2], 0, 0, 0], Q = I(12), R = I(4))
    @named quad = Quadrotor3D()

    # Use equilibrium as linearization point
    u = inputs(quad)
    x = setdiff(unknowns(quad), u)
    params = parameters(quad)
    op = Dict(vcat(x .=> x_eq, u .=> u_eq, params .=> p))

    # Linearize with ModelingToolkit
    mats, sys = linearize(quad, u, x; op)

    # Sometimes linearization will reorder the variables, but we can undo that with
    # permutation matrices Px : x_new = Px * x and Pu : u_new = Pu * u
    x_new = unknowns(sys)
    u_new = inputs(sys)

    Px = Symbolics.value.(x_new .- x') .=== 0
    Pu = Symbolics.value.(u_new .- u') .=== 0

    A_lin = Px' * mats[:A] * Px
    B_lin = Px' * mats[:B] * Pu

    K = lqr(Continuous, A_lin, B_lin, Q, R)
    return (x, _p, _t) -> -K * (x - x_eq) + u_eq
end

@named quadrotor_3d = Quadrotor3D()

# Assume rotors are negligible mass when calculating the moment of inertia
m, L = ones(2)
g = 1.0
Ixx = Iyy = m * L^2 / 6
Izz = m * L^2 / 3
Ixy = Ixz = Iyz = 0.0
p = [m, g, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]

# Create controller system and combine with quadrotor_3d, then simplify
@mtkcompile quadrotor_3d_lqr = control_quadrotor_3d(quadrotor_3d, π_lqr(p))

# Random initialization
δ = 0.5
x = get_quadrotor_3d_state_symbols(quadrotor_3d)
x0 = Dict(x .=> δ .* (2 .* rand(12) .- 1))
params = get_quadrotor_3d_param_symbols(quadrotor_3d)
p_dict = Dict(params .=> p)
op = merge(x0, p_dict)
τ = sqrt(L / g)

prob = ODEProblem(quadrotor_3d_lqr, op, 15τ)
sol = solve(prob, Tsit5())

u = get_quadrotor_3d_input_symbols(quadrotor_3d)
gif(
    plot_quadrotor_3d(
        sol,
        p;
        x_symbol=x[1],
        y_symbol=x[2],
        z_symbol=x[3],
        φ_symbol=x[4],
        θ_symbol=x[5],
        ψ_symbol=x[6],
        T_symbol=u[1],
        τφ_symbol=u[2],
        τθ_symbol=u[3],
        τψ_symbol=u[4]
    );
    fps=50
)
```

### Plotting the 3D Quadrotor

```@docs
plot_quadrotor_3d
```

### References
```@bibliography
Pages = ["quadrotor.md"]
```