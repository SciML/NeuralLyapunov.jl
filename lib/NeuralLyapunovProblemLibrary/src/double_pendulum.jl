@independent_variables t
Dt = Differential(t); DDt = Dt^2

@variables θ1(t) θ2(t) τ(t) [input = true] τ1(t) [input = true] τ2(t) [input = true]
@parameters I1 I2 l1 l2 lc1 lc2 m1 m2 g

M = [
    I1 + I2 + m2 * l1^2 + 2 * m2 * l1 * lc2 * cos(θ2)   I2 + m2 * l1 * lc2 * cos(θ2);
    I2 + m2 * l1 * lc2 * cos(θ2)                        I2
]
C = [
    -2 * m2 * l1 * lc2 * sin(θ2) * Dt(θ2)   -m2 * l1 * lc2 * sin(θ2) * Dt(θ2);
    m2 * l1 * lc2 * sin(θ2) * Dt(θ1)        0
]
G = [
    -m1 * g * lc1 * sin(θ1) - m2 * g * (l1 * sin(θ1) + lc2 * sin(θ1 + θ2));
    -m2 * g * lc2 * sin(θ1 + θ2)
]
q = [θ1; θ2]
u = [τ1; τ2]
p = [I1, I2, l1, l2, m1, m2, g]

############################## Fully-actuated double pendulum ##############################
B = [1 0; 0 1]
eqs = M * DDt.(q) + C * Dt.(q) .~ G + B * u

@named double_pendulum = ODESystem(
    eqs,
    t,
    vcat(q, u),
    p
)

########################## Acrobot (underactuated double pendulum) #########################
B = [1; 0]
eqs = M * DDt.(q) + C * Dt.(q) .~ G + B * τ

@named acrobot = ODESystem(
    eqs,
    t,
    vcat(q, τ),
    p
)

################################# Undriven double pendulum #################################
eqs = M * DDt.(q) + C * Dt.(q) .~ G

@named double_pendulum_undriven = ODESystem(
    eqs,
    t,
    q,
    p
)
