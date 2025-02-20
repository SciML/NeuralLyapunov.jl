##################################### Planar quadrotor #####################################
@independent_variables t
Dt = Differential(t); DDt = Dt^2
@variables x(t) y(t) θ(t) u1(t) u2(t)
@parameters m I g r

eqs = [
    m * DDt(x) ~ -(u1 + u2) * sin(θ);
    m * DDt(y) ~ (u1 + u2) * cos(θ) - m * g;
    I * DDt(θ) ~ r * (u1 - u2)
]

@named quadrotor_planar = ODESystem(eqs, t, [x, y, θ, u1, u2], [m, I, g, r])

####################################### 3D quadrotor #######################################
# Model from "Minimum Snap Trajectory Generation and Control for Quadrotors"
# https://doi.org/10.1109/ICRA.2011.5980409
@independent_variables t
Dt = Differential(t); DDt = Dt^2

# Position (world frame)
@variables x(t) y(t) z(t)
position_world = [x, y, z]

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
@variables T(t) [input=true] τφ(t) [input=true] τθ(t) [input=true] τψ(t) [input=true]
F = T * R[3, :]
τ = [τφ; τθ; τψ]

# Parameters
# m-mass, g-gravitational accelerationz
@parameters m g=9.81 Ixx Ixy Ixz Iyy Iyz Izz
params = [m, g, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
g_vec = [0, 0, -g]
inertia_matrix = [Ixx Ixy Ixz; Ixy Iyy Iyz; Ixz Ixy Izz]

eqs = vcat(
    DDt.(position_world) .~ F / m + g_vec,
    Dt.(angular_velocity_world) .~ inertia_matrix \
            (τ - angular_velocity_world × (inertia_matrix * angular_velocity_world)),
    Dt.(attitude) .~ inv(R) * angular_velocity_world
)

@named quadrotor_3d = ODESystem(
    eqs,
    t,
    vcat(position_world, attitude, angular_velocity_world, T, τ),
    params
)
