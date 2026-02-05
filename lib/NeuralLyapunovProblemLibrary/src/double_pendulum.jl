"""
    DoublePendulum(; actuation=:fully_actuated, name, param_defaults)

Create an `System` representing an undamped double pendulum.

The posture of the double pendulum is determined by `θ1` and `θ2`, the angle of the
first and second pendula, respectively.
`θ1` is measured counter-clockwise relative to the downward equilibrium and `θ2` is
measured counter-clockwise relative to `θ1` (i.e., when `θ2` is fixed at 0, the double
pendulum appears as a single pendulum).

The System uses the explicit manipulator form of the equations:
```math
q̈ = M^{-1}(q) (-C(q,q̇)q̇ + τ_g(q) + Bu).
```

The name of the `System` is `name`.

# Actuation modes

The four actuation modes are described in the table below and selected via `actuation`.

| Actuation mode (`actuation`) | Torque around `θ1` | Torque around `θ2` |
| ---------------------------- | ------------------ | ------------------ |
| `:fully_actuated` (default)  | `τ1`               | `τ2`               |
| `:acrobot`                   | Not actuated       | `τ`                |
| `:pendubot`                  | `τ`                | Not actuated       |
| `:undriven`                  | Not actuated       | Not actuated       |

# System Parameters
  - `I1`: moment of inertia of the first pendulum around its pivot (not its center of
    mass).
  - `I2`:  moment of inertia of the second pendulum around its pivot (not its center of
    mass).
  - `l1`: length of the first pendulum.
  - `l2`: length of the second pendulum.
  - `lc1`: distance from pivot to the center of mass of the first pendulum.
  - `lc2`: distance from the link to the center of mass of the second pendulum.
  - `m1`: mass of the first pendulum.
  - `m2`: mass of the second pendulum.
  - `g`: gravitational acceleration (defaults to 9.81).

Users may optionally provide default values of the parameters through `param_defaults`: a
vector of the default values for `[I1, I2, l1, l2, lc1, lc2, m1, m2, g]`.
"""
function DoublePendulum(; actuation = :fully_actuated, name, param_defaults = NullParameters())
    @variables θ1(t) θ2(t)
    @parameters I1 I2 l1 l2 lc1 lc2 m1 m2 g = 9.81

    M = [
        I1 + I2 + m2 * l1^2 + 2 * m2 * l1 * lc2 * cos(θ2) I2 + m2 * l1 * lc2 * cos(θ2);
        I2 + m2 * l1 * lc2 * cos(θ2) I2
    ]
    C = [
        -2 * m2 * l1 * lc2 * sin(θ2) * Dt(θ2) -m2 * l1 * lc2 * sin(θ2) * Dt(θ2);
        m2 * l1 * lc2 * sin(θ2) * Dt(θ1) 0
    ]
    G = [
        -m1 * g * lc1 * sin(θ1) - m2 * g * (l1 * sin(θ1) + lc2 * sin(θ1 + θ2));
        -m2 * g * lc2 * sin(θ1 + θ2)
    ]
    q = [θ1, θ2]
    params = [I1, I2, l1, l2, lc1, lc2, m1, m2, g]

    kwargs = if param_defaults == NullParameters()
        (; name)
    else
        (; name, initial_conditions = Dict(params .=> param_defaults))
    end

    if actuation == :fully_actuated
        ########################## Fully-actuated double pendulum ##########################
        @variables τ1(t) [input = true] τ2(t) [input = true]
        u = [τ1, τ2]

        eqs = DDt.(q) .~ M \ (-C * Dt.(q) + G + u)
        return System(eqs, t, vcat(q, u), params; kwargs...)
    elseif actuation == :undriven
        ############################# Undriven double pendulum #############################
        eqs = DDt.(q) .~ M \ (-C * Dt.(q) + G)
        return System(eqs, t, q, params; kwargs...)
    else
        ########################## Underactuated double pendulum ###########################
        @variables τ(t) [input = true]

        if actuation == :acrobot
            #################################### Acrobot ###################################
            B = [0, 1]
            eqs = DDt.(q) .~ M \ (-C * Dt.(q) + G + B * τ)

            return System(eqs, t, vcat(q, τ), params; kwargs...)
        elseif actuation == :pendubot
            ################################### Pendubot ###################################
            B = [1, 0]
            eqs = DDt.(q) .~ M \ (-C * Dt.(q) + G + B * τ)

            return System(eqs, t, vcat(q, τ), params; kwargs...)
        else
            error(
                "Invalid actuation for DoublePendulum. Received actuation = :",
                string(name)
            )
        end
    end
end

"""
    Acrobot(; name, param_defaults)

Alias for [`DoublePendulum(; actuation = :acrobot, name, param_defaults)`](@ref).
"""
function Acrobot(; name, param_defaults = NullParameters())
    return DoublePendulum(; actuation = :acrobot, name, param_defaults)
end

"""
    Pendubot(; name, param_defaults)

Alias for [`DoublePendulum(; actuation = :pendubot, name, param_defaults)`](@ref).
"""
function Pendubot(; name, param_defaults = NullParameters())
    return DoublePendulum(; actuation = :pendubot, name, param_defaults)
end

"""
    control_double_pendulum(pend, controller; name)

Control the given driven double pendulum `pend` using the provided `controller` function.

The `controller` function should have the signature `controller(x, p, t)`, where `x` is the
state vector `[θ1, θ2, ω1, ω2]`, `p` is the parameter vector
`[I1, I2, l1, l2, lc1, lc2, m1, m2, g]`, and `t` is time.
If the double pendulum has a single actuator, the controller should return a single torque.
If the double pendulum has two actuators, the controller should return a vector of two
torques `[τ1, τ2]`.
The resulting controlled pendulum system will have the name `name`.

# Example

```jldoctest; output = false
# Define a feedback cancellation controller
function π_cancellation(x, p, t)
    θ1, θ2, ω1, ω2 = x
    I1, I2, l1, l2, lc1, lc2, m1, m2, g = p
    M = [
        I1 + I2 + m2 * l1^2 + 2 * m2 * l1 * lc2 * cos(θ2) I2 + m2 * l1 * lc2 * cos(θ2);
        I2 + m2 * l1 * lc2 * cos(θ2) I2
    ]
    G = [
        -m1 * g * lc1 * sin(θ1) - m2 * g * (l1 * sin(θ1) + lc2 * sin(θ1 + θ2));
        -m2 * g * lc2 * sin(θ1 + θ2)
    ]
    return -0.1 * M \\ ([θ1, θ2] .- [π, π] + [ω1, ω2]) - G
end

# Create driven double pendulum system, apply controller, and simplify
@named double_pendulum = DoublePendulum()
@named double_pendulum_feedback_cancellation = control_double_pendulum(
    double_pendulum,
    π_cancellation
)
double_pendulum_feedback_cancellation = mtkcompile(double_pendulum_feedback_cancellation)

# Set parameter values
# Assume uniform rods of random mass and length
m1, m2 = ones(2)
l1, l2 = ones(2)
lc1, lc2 = l1 / 2, l2 / 2
I1 = m1 * l1^2 / 3
I2 = m2 * l2^2 / 3
g = 1.0
params = get_double_pendulum_param_symbols(double_pendulum)
p = Dict(params .=> [I1, I2, l1, l2, lc1, lc2, m1, m2, g])

# Construct ODE problem
x = get_double_pendulum_state_symbols(double_pendulum)
x0 = Dict(x .=> zeros(4))
t_end = 100
op = merge(x0, p)
prob = ODEProblem(double_pendulum_feedback_cancellation, op, t_end)
# output
ODEProblem with uType Vector{Float64} and tType Int64. In-place: true
Initialization status: FULLY_DETERMINED
Non-trivial mass matrix: false
timespan: (0, 100)
u0: 4-element Vector{Float64}:
 0.0
 0.0
 0.0
 0.0
```
"""
function control_double_pendulum(pend, controller; name)
    τ = unbound_inputs(pend)
    if length(τ) == 2
        τ = [pend.τ1, pend.τ2]
    elseif length(τ) == 1
        τ = [pend.τ]
    elseif length(τ) == 0
        error("Cannot control an undriven double pendulum.")
    else
        error("Unexpected number of inputs in double pendulum.")
    end

    q = [pend.θ1, pend.θ2]
    x = vcat(q, Dt.(q))
    p = [pend.I1, pend.I2, pend.l1, pend.l2, pend.lc1, pend.lc2, pend.m1, pend.m2, pend.g]

    eqs = τ .~ controller(x, p, t)

    controller_sys = System(eqs, t, q, []; name = Symbol(name, :_controller))
    return compose(controller_sys, pend; name)
end

"""
    get_double_pendulum_state_symbols(pend)

Get the state variable symbols of the given double pendulum `pend` as a vector:
`[θ1, θ2, ω1, ω2]`, where ``ω_i = \\dot{θ}_i``.
"""
function get_double_pendulum_state_symbols(pend)
    q = [pend.θ1, pend.θ2]
    return vcat(q, Dt.(q))
end

"""
    get_double_pendulum_param_symbols(pend)

Get the parameter symbols of the given double pendulum `pend` as a vector:
`[I1, I2, l1, l2, lc1, lc2, m1, m2, g]`.
"""
function get_double_pendulum_param_symbols(pend)
    return [pend.I1, pend.I2, pend.l1, pend.l2, pend.lc1, pend.lc2, pend.m1, pend.m2, pend.g]
end

"""
    plot_double_pendulum(θ1, θ2, p, t; title)
    plot_double_pendulum(sol, p; title, N, angle1_symbol, angle2_symbol)

Plot the pendulum's trajectory.

# Arguments
  - `θ1`: The angle of the first pendulum link at each time step.
  - `θ2`: The angle of the second pendulum link at each time step.
  - `t`: The time steps.
  - `sol`: The solution to the ODE problem.
  - `p`: The parameters of the double pendulum.

# Keyword arguments
  - `title`: The title of the plot; defaults to no title (i.e., `title=""`).
  - `N`: The number of points to plot; when using `θ` and `t`, uses `length(t)`; defaults to
    500 when using `sol`.
  - `angle1_symbol`: The symbol of the angle of the first link in `sol`; defaults to `:θ1`.
  - `angle2_symbol`: The symbol of the angle of the second link in `sol`; defaults to `:θ2`.
"""
function plot_double_pendulum end
