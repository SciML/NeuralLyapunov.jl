"""
    Pendulum(; driven = true, name, param_defaults)

Create an `System` representing a damped, driven or undriven pendulum, depending on the
value of `driven` (defaults to `true`, i.e., driven pendulum).

The equation used in this model is
``\\ddot{θ} + 2 ζ ω_0 \\dot{θ} + ω_0^2 \\sin(θ) = τ,``
where ``θ`` is the angle counter-clockwise from the downward equilibrium, ``ζ`` is the
damping parameter, ``ω_0`` is the resonant angular frequency, and ``τ`` is the input torque
divided by the moment of inertia of the pendulum around the pivot (`driven = false` sets
``τ ≡ 0``).

The name of the `System` is `name`.

Users may optionally provide default values of the parameters through `param_defaults`: a
vector of the default values for `[ζ, ω_0]`.

# Example

```jldoctest; output = false
@named pendulum = Pendulum(driven = false)
pendulum = mtkcompile(pendulum)

x0 = [2.0, 0.0]
p = ones(2)
t_end = 10
x = get_pendulum_state_symbols(pendulum)
params = get_pendulum_param_symbols(pendulum)

op = Dict(vcat(x, params) .=> vcat(x0, p))
prob = ODEProblem(pendulum, op, t_end)
# output
ODEProblem with uType Vector{Float64} and tType Int64. In-place: true
Initialization status: FULLY_DETERMINED
Non-trivial mass matrix: false
timespan: (0, 10)
u0: 2-element Vector{Float64}:
 2.0
 0.0
```
"""
function Pendulum(; driven = true, name, param_defaults = NullParameters())
    @variables θ(t) τ(t) [input = true]
    @parameters ζ ω_0

    params = [ζ, ω_0]
    kwargs = if param_defaults == NullParameters()
        (; name)
    else
        (; name, initial_conditions = Dict(params .=> param_defaults))
    end

    torque = if driven
        τ
    else
        0
    end
    variables = if driven
        [θ, τ]
    else
        [θ]
    end

    eqs = [DDt(θ) + 2ζ * ω_0 * Dt(θ) + ω_0^2 * sin(θ) ~ torque]

    return System(
        eqs,
        t,
        variables,
        params;
        kwargs...
    )
end

"""
    control_pendulum(pend, controller; name)

Control the given driven pendulum `pend` using the provided `controller` function.

The `controller` function should have the signature `controller(x, p, t)`, where `x` is the
state vector `[θ, ω]`, `p` is the parameter vector `[ζ, ω_0]`, and `t` is time.
The function should return the torque `τ` to be applied to the pendulum.
The resulting controlled pendulum system will have the name `name`.

# Example

```jldoctest; output = false
# Define a simple feedback cancellation controller
π_cancellation(x, p, t) = 2 * p[2]^2 * sin(x[1])

# Create driven pendulum system, apply controller, and simplify
@named pendulum = Pendulum(driven = true)
@named pendulum_feedback_cancellation = control_pendulum(pendulum, π_cancellation)
pendulum_feedback_cancellation = mtkcompile(pendulum_feedback_cancellation)

# Construct ODE problem
x0 = zeros(2)
p = ones(2)
t_end = 10
x = get_pendulum_state_symbols(pendulum)
params = get_pendulum_param_symbols(pendulum)

op = Dict(vcat(x, params) .=> vcat(x0, p))
prob = ODEProblem(pendulum_feedback_cancellation, op, t_end)
# output
ODEProblem with uType Vector{Float64} and tType Int64. In-place: true
Initialization status: FULLY_DETERMINED
Non-trivial mass matrix: false
timespan: (0, 10)
u0: 2-element Vector{Float64}:
 0.0
 0.0
```
"""
function control_pendulum(pend, controller; name)
    x = get_pendulum_state_symbols(pend)
    p = get_pendulum_param_symbols(pend)
    eqs = [pend.τ ~ controller(x, p, t)]

    controller_sys = System(eqs, t, [pend.θ], []; name = Symbol(name, :_controller))
    return compose(controller_sys, pend; name)
end

"""
    get_pendulum_state_symbols(pend)

Get the state variable symbols of the given pendulum `pend` as a vector: `[θ, ω]`, where
``ω = \\dot{θ}``.
"""
function get_pendulum_state_symbols(pend)
    θ = pend.θ
    return [θ, Dt(θ)]
end

"""
    get_pendulum_param_symbols(pend)

Get the parameter symbols of the given pendulum `pend` as a vector: `[ζ, ω_0]`.
"""
function get_pendulum_param_symbols(pend)
    return [pend.ζ, pend.ω_0]
end

"""
    plot_pendulum(θ, t; title)
    plot_pendulum(sol; title, N, angle_symbol)

Plot the pendulum's trajectory.

# Arguments
  - `θ`: The angle of the pendulum at each time step.
  - `t`: The time steps.
  - `sol`: The solution to the ODE problem.

# Keyword arguments
  - `title`: The title of the plot; defaults to no title (i.e., `title=""`).
  - `N`: The number of points to plot; when using `θ` and `t`, uses `length(t)`; defaults to
    500 when using `sol`.
  - `angle_symbol`: The symbol of the angle in `sol`; defaults to `:θ`.
"""
function plot_pendulum end
