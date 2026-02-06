"""
    QuadrotorPlanar(; name, defaults)

Create an `System` representing a planar approximation of the quadrotor (technically a
birotor).

This birotor is a rigid body with two rotors in line with the center of mass.
The location of the center of mass is determined by `x` and `y`.
Its orientation is determined by `θ`, measured counter-clockwise from the ``x``-axis.
The thrust from the right rotor (on the positive ``x``-axis when ``θ = 0``) is the input
`u1`.
The thrust from the other rotor is `u2`.
Note that these thrusts should be nonnegative and if a negative input is provided, the model
replaces it with 0.

The equations governing the planar quadrotor are:
```math
\\begin{align}
    mẍ &= -(u_1 + u_2)\\sin(θ), \\\\
    mÿ &= (u_1 + u_2)\\cos(θ) - mg, \\\\
    I_{quad} \\ddot{θ} &= r (u_1 - u_2).
\\end{align}
```

The name of the `System` is `name`.

# System Parameters
  - `m`: mass of the quadrotor.
  - `I_quad`: moment of inertia of the quadrotor around its center of mass.
  - `g`: gravitational acceleration in the direction of the negative ``y``-axis (defaults to
    9.81).
  - `r`: distance from center of mass to each rotor.

Users may optionally provide default values of the parameters through `defaults`: a
vector of the default values for `[m, I_quad, g, r]`.
"""
function QuadrotorPlanar(; name, defaults = NullParameters())
    @variables x(t) y(t) θ(t)
    @variables u1(t) [input = true] u2(t) [input = true]
    @parameters m I_quad g r

    # Thrusts must be nonnegative
    ũ1 = max(0, u1)
    ũ2 = max(0, u2)

    eqs = [
        m * DDt(x) ~ -(ũ1 + ũ2) * sin(θ);
        m * DDt(y) ~ (ũ1 + ũ2) * cos(θ) - m * g;
        I_quad * DDt(θ) ~ r * (ũ1 - ũ2)
    ]

    params = [m, I_quad, g, r]
    kwargs = if defaults == NullParameters()
        (; name)
    else
        (; name, initial_conditions = Dict(params .=> defaults))
    end

    return System(eqs, t, [x, y, θ, u1, u2], params; kwargs...)
end

"""
    control_quadrotor_planar(quadrotor, controller; name)

Control the given planar quadrotor `quadrotor` using the provided `controller` function.

The `controller` function should have the signature `controller(x, p, t)`, where `x` is the
state vector `[x, y, θ, vx, vy, ω]`, `p` is the parameter vector `[m, I_quad, g, r]`, and
`t` is time.
The function should return the thrusts `[u1, u2]` to be applied to the quadrotor (both
nonnegative).

The resulting controlled quadrotor system will have the name `name`.
"""
function control_quadrotor_planar(quadrotor, controller; name)
    q = [quadrotor.x, quadrotor.y, quadrotor.θ]
    x = vcat(q, Dt.(q))
    p = [quadrotor.m, quadrotor.I_quad, quadrotor.g, quadrotor.r]
    u = [quadrotor.u1, quadrotor.u2]

    eqs = u .~ controller(x, p, t)

    controller_sys = System(eqs, t, q, []; name = Symbol(name, :_controller))
    return compose(controller_sys, quadrotor; name)
end

"""
    get_quadrotor_planar_state_symbols(quadrotor)

Get the state variable symbols of the given planar quadrotor `quadrotor` as a vector:
`[x, y, θ, vx, vy, ω]`, where ``vx = \\dot{x}``, ``vy = \\dot{y}``, and ``ω = \\dot{θ}``.
"""
function get_quadrotor_planar_state_symbols(quadrotor)
    q = [quadrotor.x, quadrotor.y, quadrotor.θ]
    return vcat(q, Dt.(q))
end

"""
    get_quadrotor_planar_param_symbols(quadrotor)

Get the parameter symbols of the given planar quadrotor `quadrotor` as a vector:
`[m, I_quad, g, r]`.
"""
function get_quadrotor_planar_param_symbols(quadrotor)
    return [quadrotor.m, quadrotor.I_quad, quadrotor.g, quadrotor.r]
end

"""
    get_quadrotor_planar_input_symbols(quadrotor)

Get the input variable symbols of the given planar quadrotor `quadrotor` as a vector:
`[u1, u2]`.
"""
function get_quadrotor_planar_input_symbols(quadrotor)
    return [quadrotor.u1, quadrotor.u2]
end

"""
    plot_quadrotor_planar(x, y, θ, [u1, u2,] p, t; title)
    plot_quadrotor_planar(sol, p; title, N, x_symbol, y_symbol, θ_symbol)

Plot the planar quadrotor's trajectory.

When thrusts are supplied, the arrows scale with thrust, otherwise the arrows are of
constant length.

# Arguments
  - `x`: The x-coordinate of the quadrotor at each time step.
  - `y`: The y-coordinate of the quadrotor at each time step.
  - `θ`: The angle of the quadrotor at each time step.
  - `u1`: The thrust of the first rotor at each time step.
  - `u2`: The thrust of the second rotor at each time step.
  - `t`: The time steps.
  - `sol`: The solution to the ODE problem.
  - `p`: The parameters of the quadrotor.

# Keyword arguments
  - `title`: The title of the plot; defaults to no title (i.e., `title=""`).
  - `N`: The number of points to plot; when using `x`, `y`, `θ`, and `t`, uses `length(t)`; defaults to
    500 when using `sol`.
  - `x_symbol`: The symbol of the x-coordinate in `sol`; defaults to `:x`.
  - `y_symbol`: The symbol of the y-coordinate in `sol`; defaults to `:y`.
  - `θ_symbol`: The symbol of the angle in `sol`; defaults to `:θ`.
  - `u1_symbol`: The symbol of the thrust of the first rotor in `sol`; defaults to `:u1`.
  - `u2_symbol`: The symbol of the thrust of the second rotor in `sol`; defaults to `:u2`.
"""
function plot_quadrotor_planar end
