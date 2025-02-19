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
position_world = [x; y; z]

# Attitude
# φ-roll (around body x-axis), θ-pitch (around body y-axis), ψ-yaw (around body z-axis)
@variables φ(t) θ(t) ψ(t)
attitude = [φ, θ, ψ]
angular_velocity_body = Dt.(attitude)
R = RotZXY(roll=φ, pitch=θ, yaw=ψ)
angular_velocity_world = R * angular_velocity_body

# Inputs
# T-thrust, τφ-roll torque, τθ-pitch torque, τψ-yaw torque
@variables T(t) [input=true] τφ(t) [input=true] τθ(t) [input=true] τψ(t) [input=true]
F = T * R[3, :]
τ = [τφ; τθ; τψ]

# Parameters
# m-mass, Ix-moment of inertia about body x-axis, Iy-moment of inertia about body y-axis,
# Iz-moment of inertia about body z-axis, g-gravitational acceleration, l-distance from
# center of mass to rotor
@parameters m g I11 I12 I13 I21 I22 I23 I31 I32 I33
params = [m, g, I11, I12, I13, I21, I22, I23, I31, I32, I33]
g_vec = [0; 0; -g]
I = [I11 I12 I13; I21 I22 I23; I31 I32 I33]

eqs = vcat(
    DDt.(position_world) .~ F / m + g_vec,
    I * Dt.(angular_velocity_world) .~
            τ - angular_velocity_world × (I * angular_velocity_world)
)

@named quadrotor_3d = ODESystem(
    eqs,
    t,
    vcat(position_world, attitude, T, τ),
    params
)
