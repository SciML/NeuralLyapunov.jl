##################################### Planar quadrotor #####################################
@independent_variables t
Dt = Differential(t); DDt = Dt^2
@variables x(t) y(t) θ(t)
@variables u1(t) [input=true] u2(t) [input=true]
@parameters m I_quad g r

# Thrusts must be nonnegative
ũ1 = max(0, u1)
ũ2 = max(0, u2)

eqs = [
    m * DDt(x) ~ -(ũ1 + ũ2) * sin(θ);
    m * DDt(y) ~ (ũ1 + ũ2) * cos(θ) - m * g;
    I_quad * DDt(θ) ~ r * (ũ1 - ũ2)
]

@named quadrotor_planar = ODESystem(eqs, t, [x, y, θ, u1, u2], [m, I_quad, g, r])

####################################### 3D quadrotor #######################################
# Model from "Minimum Snap Trajectory Generation and Control for Quadrotors"
# https://doi.org/10.1109/ICRA.2011.5980409
@independent_variables t
Dt = Differential(t); DDt = Dt^2

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
R = RotZXY(roll=φ, pitch=θ, yaw=ψ)

# Angular velocity (world frame)
@variables ωφ(t), ωθ(t), ωψ(t)
angular_velocity_world = [ωφ, ωθ, ωψ]

# Inputs
# T-thrust, τφ-roll torque, τθ-pitch torque, τψ-yaw torque
@variables T(t) [input=true]
@variables τφ(t) [input=true] τθ(t) [input=true] τψ(t) [input=true]

# Individual rotor thrusts must be nonnegative
f = ([1 0 -2 -1; 1 2 0 1; 1 0 2 -1; 1 -2 0 1] * [T; τφ; τθ; τψ]) ./ 4
f̃ = max.(0, f)
T̃ = [1 1 1 1; 0 1 0 -1; -1 0 1 0; -1 1 -1 1] * f̃

F = T̃[1] .* R[3, :]
τ = T̃[2:4]

# Parameters
# m-mass, g-gravitational accelerationz
@parameters m g=9.81 Ixx Ixy Ixz Iyy Iyz Izz
params = [m, g, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
g_vec = [0, 0, -g]
inertia_matrix = [Ixx Ixy Ixz; Ixy Iyy Iyz; Ixz Ixy Izz]

eqs = vcat(
    Dt.(position_world) .~ velocity_world,
    Dt.(velocity_world) .~ F ./ m .+ g_vec,
    Dt.(attitude) .~ inv(R) * angular_velocity_world,
    Dt.(angular_velocity_world) .~ inertia_matrix \
            (τ - angular_velocity_world × (inertia_matrix * angular_velocity_world))
)

@named quadrotor_3d = ODESystem(
    eqs,
    t,
    vcat(position_world, attitude, velocity_world, angular_velocity_world, T, τφ, τθ, τψ),
    params
)

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
