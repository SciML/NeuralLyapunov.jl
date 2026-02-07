"""
    Quadrotor3D(; name, defaults)

Create an `System` representing a quadrotor in 3D space.

The quadrotor is a rigid body in an X-shape (90°-angles between the rotors).
The equations governing the quadrotor can be found in [quadrotor](@cite).

# System State Variables
  - `x`: ``x``-position (world frame).
  - `y`: ``y``-position (world frame).
  - `z`: ``z``-position (world frame).
  - `φ`: roll around body ``x``-axis (Z-X-Y Euler angles).
  - `θ`: pitch around body ``y``-axis (Z-X-Y Euler angles).
  - `ψ`: yaw around body ``z``-axis (Z-X-Y Euler angles).
  - `vx`: ``x``-velocity (world frame).
  - `vy`: ``y``-velocity (world frame).
  - `vz`: ``z``-velocity (world frame).
  - `ωφ`: roll angular velocity (world frame).
  - `ωθ`: pitch angular velocity (world frame).
  - `ωψ`: yaw angular velocity (world frame).

# System Input Variables
  - `T`: thrust (should be nonnegative).
  - `τφ`: roll torque.
  - `τθ`: pitch torque.
  - `τψ`: yaw torque.

Not only should the aggregate thrust be nonnegative, but the torques should have been
generated from nonnegative individual rotor thrusts.
The model calculates individual rotor thrusts and replaces any negative values with 0.

# System Parameters
  - `m`: mass of the quadrotor.
  - `g`: gravitational acceleration in the direction of the negative ``z``-axis (defaults to
    9.81).
  - `Ixx`,`Ixy`,`Ixz`,`Iyy`,`Iyz`,`Izz`: components of the moment of inertia matrix of the
    quadrotor around its center of mass:
    ```math
    I = \\begin{pmatrix}
            I_{xx} & I_{xy} & I_{xz} \\\\
            I_{xy} & I_{yy} & I_{yz} \\\\
            I_{xz} & I_{yz} & I_{zz}
        \\end{pmatrix}.
    ```

Users may optionally provide default values of the parameters through `defaults`: a
vector of the default values for `[m, g, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]`.
"""
function Quadrotor3D(; name, defaults = NullParameters())
    # Model from "Minimum Snap Trajectory Generation and Control for Quadrotors"
    # https://doi.org/10.1109/ICRA.2011.5980409

    # Position (world frame)
    @variables x(t) y(t) z(t)
    position_world = [x, y, z]

    # Velocity (world frame)
    @variables vx(t) vy(t) vz(t)
    velocity_world = [vx, vy, vz]

    # Attitude
    # φ-roll (around body x-axis), θ-pitch (around body y-axis), ψ-yaw (around body z-axis)
    @variables φ(t) θ(t) ψ(t)
    attitude = [φ, θ, ψ]
    R = RotZXY(roll = φ, pitch = θ, yaw = ψ)

    # Angular velocity (world frame)
    @variables ωφ(t), ωθ(t), ωψ(t)
    ω_world = [ωφ, ωθ, ωψ]

    # Inputs
    # T-thrust, τφ-roll torque, τθ-pitch torque, τψ-yaw torque
    @variables T(t) [input = true]
    @variables τφ(t) [input = true] τθ(t) [input = true] τψ(t) [input = true]

    # Individual rotor thrusts must be nonnegative
    f = ([1 0 -2 -1; 1 2 0 1; 1 0 2 -1; 1 -2 0 1] * [T; τφ; τθ; τψ]) ./ 4
    f̃ = max.(0, f)
    T̃ = [1 1 1 1; 0 1 0 -1; -1 0 1 0; -1 1 -1 1] * f̃

    F = T̃[1] .* R[3, :]
    τ = T̃[2:4]

    # Parameters
    # m-mass, g-gravitational accelerationz
    @parameters m g = 9.81 Ixx Ixy Ixz Iyy Iyz Izz
    params = [m, g, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
    g_vec = [0, 0, -g]
    inertia_matrix = [Ixx Ixy Ixz; Ixy Iyy Iyz; Ixz Ixy Izz]

    eqs = vcat(
        Dt.(position_world) .~ velocity_world,
        Dt.(velocity_world) .~ F ./ m .+ g_vec,
        Dt.(attitude) .~ inv(R) * ω_world,
        Dt.(ω_world) .~ inertia_matrix \ (τ - ω_world × (inertia_matrix * ω_world))
    )

    kwargs = if defaults == NullParameters()
        (; name)
    else
        (; name, initial_conditions = Dict(params .=> defaults))
    end

    return System(
        eqs,
        t,
        vcat(position_world, attitude, velocity_world, ω_world, T, τφ, τθ, τψ),
        params;
        kwargs...
    )
end

"""
    control_quadrotor_3d(quadrotor, controller; name)

Control the given 3D quadrotor `quadrotor` using the provided `controller` function.

The `controller` function should have the signature `controller(x, p, t)`, where `x` is the
state vector `[x, y, z, φ, θ, ψ, vx, vy, vz, ωφ, ωθ, ωψ]`, `p` is the parameter vector
`[m, g, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]`, and `t` is time.
The function should return the thrust and torques `[T, τφ, τθ, τψ]` to be applied to the
quadrotor.
Not only should the aggregate thrust be nonnegative, but the torques should have been
generated from nonnegative individual rotor thrusts.
See [`Quadrotor3D`](@ref) for more model details.

The resulting controlled quadrotor system will have the name `name`.
"""
function control_quadrotor_3d(quadrotor, controller; name)
    position_world = [quadrotor.x, quadrotor.y, quadrotor.z]
    attitude = [quadrotor.φ, quadrotor.θ, quadrotor.ψ]
    velocity_world = [quadrotor.vx, quadrotor.vy, quadrotor.vz]
    ω_world = [quadrotor.ωφ, quadrotor.ωθ, quadrotor.ωψ]
    x = vcat(position_world, attitude, velocity_world, ω_world)

    p = [
        quadrotor.m,
        quadrotor.g,
        quadrotor.Ixx,
        quadrotor.Ixy,
        quadrotor.Ixz,
        quadrotor.Iyy,
        quadrotor.Iyz,
        quadrotor.Izz,
    ]
    u = [quadrotor.T, quadrotor.τφ, quadrotor.τθ, quadrotor.τψ]

    eqs = u .~ controller(x, p, t)

    controller_sys = System(eqs, t, x, []; name = Symbol(name, :_controller))
    return compose(controller_sys, quadrotor; name)
end

"""
    get_quadrotor_3d_state_symbols(quadrotor)

Get the state variable symbols of the given 3D quadrotor `quadrotor` as a vector:
`[x, y, z, φ, θ, ψ, vx, vy, vz, ωφ, ωθ, ωψ]`.
"""
function get_quadrotor_3d_state_symbols(quadrotor)
    position_world = [quadrotor.x, quadrotor.y, quadrotor.z]
    attitude = [quadrotor.φ, quadrotor.θ, quadrotor.ψ]
    velocity_world = [quadrotor.vx, quadrotor.vy, quadrotor.vz]
    ω_world = [quadrotor.ωφ, quadrotor.ωθ, quadrotor.ωψ]
    return vcat(position_world, attitude, velocity_world, ω_world)
end

"""
    get_quadrotor_3d_param_symbols(quadrotor)

Get the parameter symbols of the given 3D quadrotor `quadrotor` as a vector:
`[m, g, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]`.
"""
function get_quadrotor_3d_param_symbols(quadrotor)
    return [
        quadrotor.m,
        quadrotor.g,
        quadrotor.Ixx,
        quadrotor.Ixy,
        quadrotor.Ixz,
        quadrotor.Iyy,
        quadrotor.Iyz,
        quadrotor.Izz,
    ]
end

"""
    get_quadrotor_3d_input_symbols(quadrotor)

Get the input variable symbols of the given 3D quadrotor `quadrotor` as a vector:
`[T, τφ, τθ, τψ]`.
"""
function get_quadrotor_3d_input_symbols(quadrotor)
    return [quadrotor.T, quadrotor.τφ, quadrotor.τθ, quadrotor.τψ]
end

"""
    plot_quadrotor_3d(x, y, z, φ, θ, ψ, [T, τφ, τθ, τψ,] p, t; title)
    plot_quadrotor_3d(sol, p; title, N, x_symbol, y_symbol, z_symbol, φ_symbol, θ_symbol, ψ_symbol, T_symbol, τφ_symbol, τθ_symbol, τψ_symbol)

Plot the 3D quadrotor's trajectory.

When thrusts are supplied, the arrows scale with thrust, otherwise the arrows are of
constant length.

# Arguments
  - `x`: The x-coordinate of the quadrotor at each time step.
  - `y`: The y-coordinate of the quadrotor at each time step.
  - `z`: The z-coordinate of the quadrotor at each time step.
  - `φ`: The roll of the quadrotor at each time step.
  - `θ`: The pitch of the quadrotor at each time step.
  - `ψ`: The yaw of the quadrotor at each time step.
  - `T`: The thrust of the quadrotor at each time step.
  - `τφ`: The roll torque of the quadrotor at each time step.
  - `τθ`: The pitch torque of the quadrotor at each time step.
  - `τψ`: The yaw torque of the quadrotor at each time step.
  - `t`: The time steps.
  - `sol`: The solution to the ODE problem.
  - `p`: The parameters of the quadrotor.

# Keyword arguments
  - `title`: The title of the plot; defaults to no title (i.e., `title=""`).
  - `N`: The number of points to plot; when using `x`, `y`, `z`, etc., uses `length(t)`;
    defaults to 500 when using `sol`.
  - `x_symbol`: The symbol of the x-coordinate in `sol`; defaults to `:x`.
  - `y_symbol`: The symbol of the y-coordinate in `sol`; defaults to `:y`.
  - `z_symbol`: The symbol of the z-coordinate in `sol`; defaults to `:z`.
  - `φ_symbol`: The symbol of the roll in `sol`; defaults to `:φ`.
  - `θ_symbol`: The symbol of the pitch in `sol`; defaults to `:θ`.
  - `ψ_symbol`: The symbol of the yaw in `sol`; defaults to `:ψ`.
  - `T_symbol`: The symbol of the thrust in `sol`; defaults to `:T`.
  - `τφ_symbol`: The symbol of the roll torque in `sol`; defaults to `:τφ`.
  - `τθ_symbol`: The symbol of the pitch torque in `sol`; defaults to `:τθ`.
  - `τψ_symbol`: The symbol of the yaw torque in `sol`; defaults to `:τψ`.
"""
function plot_quadrotor_3d end
