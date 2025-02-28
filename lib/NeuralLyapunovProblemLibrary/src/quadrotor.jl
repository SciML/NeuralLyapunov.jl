##################################### Planar quadrotor #####################################
@independent_variables t
Dt = Differential(t); DDt = Dt^2
@variables x(t) y(t) θ(t)
@variables u1(t) [input=true] u2(t) [input=true]
@parameters m I_quad g r

# Thrusts must be positive
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

# Individual rotor thrusts must be positive
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

function plot_quadrotor_planar end
function plot_quadrotor_3d end
