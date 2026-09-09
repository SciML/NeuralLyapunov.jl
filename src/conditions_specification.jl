"""
    AbstractNeuralLyapunovStructure{nc}

Developer interface for a neural Lyapunov-function structure.

The type parameter `nc` records whether the structure includes a neural-network-dependent
control or other contribution to the dynamics. A subtype must implement the following
generic functions for every instance:

  - [`get_V`](@ref): return `V(phi, state, fixed_point)`, where `phi` is a callable neural
    network and `state` is a single state vector.
  - [`get_V̇`](@ref): return `V̇(phi, J_phi, state, dstate_dt, fixed_point)`, where `J_phi`
    evaluates the derivative of `phi` with respect to `state`.
  - [`get_network_dim`](@ref): return the positive number of neural-network outputs used by
    the structure.

For a subtype of `AbstractNeuralLyapunovStructure{true}`, also implement
[`get_control_structure`](@ref), returning `control_structure(phi_c, state, fixed_point)`,
and [`get_control_dim`](@ref), returning the positive number of outputs consumed by that
structure. `phi` and `J_phi` must be functions of `state` alone; implementations must not
assume a particular concrete structure type or access its fields from generic code.

The `false` and `true` parameter values are part of the dispatch contract. Use
`AbstractNeuralLyapunovStructure{false}` for dynamics of the form `f(state, p, t)` and
`AbstractNeuralLyapunovStructure{true}` for dynamics of the form `f(state, control, p, t)`.

# Example

```julia
struct MyStructure <: AbstractNeuralLyapunovStructure{false} end

NeuralLyapunov.get_V(::MyStructure) = (phi, state, fixed_point) -> phi(state)[1]
NeuralLyapunov.get_V̇(::MyStructure) =
    (phi, J_phi, state, dstate_dt, fixed_point) -> sum(J_phi(state)[1, :] .* dstate_dt)
NeuralLyapunov.get_network_dim(::MyStructure) = 1
```
"""
abstract type AbstractNeuralLyapunovStructure{nc} end

"""
    AbstractLyapunovMinimizationCondition

Developer interface for the minimization condition in a neural Lyapunov problem.

A subtype must implement [`check_nonnegativity`](@ref),
[`check_minimal_fixed_point`](@ref), and [`get_minimization_condition`](@ref). The first two
generic functions return `Bool`s. If `check_nonnegativity(cond)` is `true`,
`get_minimization_condition(cond)` must return a callable
`(V, state, fixed_point) -> residual`, where `V` is callable as `V(state)` and `residual`
is a scalar or a vector whose entries are all zero when the condition is satisfied. If the
flag is `false`, the returned condition may be `nothing`, because no minimization equation
is generated. The `check_minimal_fixed_point` flag independently controls whether
`V(fixed_point) = 0` is added by [`NeuralLyapunovPDESystem`](@ref).

Implementations should extend these generic functions for their own subtype and should not
depend on the fields of another package's concrete condition.

# Example

```julia
struct MyMinimizationCondition <: AbstractLyapunovMinimizationCondition end

NeuralLyapunov.check_nonnegativity(::MyMinimizationCondition) = true
NeuralLyapunov.check_minimal_fixed_point(::MyMinimizationCondition) = true
NeuralLyapunov.get_minimization_condition(::MyMinimizationCondition) =
    (V, state, fixed_point) -> V(state) - V(fixed_point)
```
"""
abstract type AbstractLyapunovMinimizationCondition end

"""
    AbstractLyapunovDecreaseCondition

Developer interface for the decrease condition in a neural Lyapunov problem.

A subtype must implement [`check_decrease`](@ref) and [`get_decrease_condition`](@ref).
`check_decrease(cond)` returns a `Bool`. If it is `true`,
`get_decrease_condition(cond)` must return a callable
`(V, V̇, state, fixed_point) -> residual`, where `V` and `V̇` are callable at `state` and the
residual is a scalar or vector that is zero when the condition is satisfied. If the flag is
`false`, the returned condition may be `nothing`, because no decrease equation is generated.
The implementation may evaluate `V` and `V̇` at additional points, but it must preserve this
call signature so that it works with the generic PDESystem construction.

Implementations should extend these generic functions for their own subtype and should not
depend on the fields of another package's concrete condition.

# Example

```julia
struct MyDecreaseCondition <: AbstractLyapunovDecreaseCondition end

NeuralLyapunov.check_decrease(::MyDecreaseCondition) = true
NeuralLyapunov.get_decrease_condition(::MyDecreaseCondition) =
    (V, V̇, state, fixed_point) -> V̇(state)
```
"""
abstract type AbstractLyapunovDecreaseCondition end

"""
    NeuralLyapunovSpecification(structure, minimization_condition, decrease_condition)

Specifies a neural Lyapunov problem.

# Fields
  - `structure`: a [`NeuralLyapunovStructure`](@ref) specifying the relationship between the
    neural network and the candidate Lyapunov function.
  - `minimization_condition`: an [`AbstractLyapunovMinimizationCondition`](@ref) specifying
    how the minimization condition will be enforced.
  - `decrease_condition`: an [`AbstractLyapunovDecreaseCondition`](@ref) specifying how the
    decrease condition will be enforced.

# Example
```julia
julia> NeuralLyapunovSpecification(NonnegativeStructure(1), PositiveSemiDefinite(), StabilityISL())
NeuralLyapunovSpecification
    Structure:
        NeuralLyapunovStructure
            Network dimension: 1
            V(x) = φ(x)²
            V̇(x) = 2ẋ*∇φ(x)*φ(x)
    Minimization Condition:
        LyapunovMinimizationCondition
            Trains for V(x) ≥ 0.0
            with approximation a ≤ 0 => max(0, a) ≈ 0
            Trains for V(x_0) = 0
    Decrease Condition:
        LyapunovDecreaseCondition
            Trains for V̇(x) ≤ 0
            with approximation a ≤ 0 => max(0, a) ≈ 0
```
"""
struct NeuralLyapunovSpecification
    structure::AbstractNeuralLyapunovStructure
    minimization_condition::AbstractLyapunovMinimizationCondition
    decrease_condition::AbstractLyapunovDecreaseCondition
end

function Base.show(io::IO, spec::NeuralLyapunovSpecification)
    # Regex indents all nonempty lines by 8 spaces
    println(io, "NeuralLyapunovSpecification")
    println(io, "    Structure:")
    println(io, replace(string(spec.structure), r"^(?=.)"m => "        "))
    println(io, "    Minimization Condition:")
    println(io, replace(string(spec.minimization_condition), r"^(?=.)"m => "        "))
    println(io, "    Decrease Condition:")
    print(io, replace(string(spec.decrease_condition), r"^(?=.)"m => "        ", r"(ẋ|ẋ)" => "ẋ"))
    return
end

"""
    get_V(str::AbstractNeuralLyapunovStructure)

Return a function `V(phi, state, fixed_point)` that outputs the value of the Lyapunov
function at `state`.

# Arguments

- `str`: the neural Lyapunov structure being queried.

# Returns

A callable accepting a neural network `phi`, a state vector, and a fixed point. It may
return a scalar or an array compatible with the condition and PDESystem constructors.

# Extension Rules

Define a method for each concrete subtype. The returned callable must not rely on a
specific neural-network implementation.
"""
function get_V(str::AbstractNeuralLyapunovStructure)
    error(
        "get_V not implemented for AbstractNeuralLyapunovStructure of type " *
            string(typeof(str)) * "."
    )
end

"""
    get_V̇(str::AbstractNeuralLyapunovStructure)

Return a function `V̇(phi, J_phi, state, dstate_dt, fixed_point)` that outputs the
time derivative of the Lyapunov function at `state`.

# Arguments

- `str`: the neural Lyapunov structure being queried.

# Returns

A callable accepting the neural network `phi`, its state Jacobian `J_phi`, a state vector,
the state derivative `dstate_dt`, and a fixed point. It may return a scalar or an array
compatible with the generated equations.

# Extension Rules

Define a method for each concrete subtype. `J_phi` must be treated as a callable of the
state, rather than as a package-specific differentiation object.
"""
function get_V̇(str::AbstractNeuralLyapunovStructure)
    error(
        "get_V̇ not implemented for AbstractNeuralLyapunovStructure of type " *
            string(typeof(str)) * "."
    )
end

"""
    get_network_dim(str::AbstractNeuralLyapunovStructure)

Return the number of dimensions of the neural network output specified by `str`.

# Arguments

- `str`: the neural Lyapunov structure being queried.

# Returns

A positive `Integer` equal to the number of neural-network outputs consumed by `get_V` and
`get_V̇`.
"""
function get_network_dim(str::AbstractNeuralLyapunovStructure)
    error(
        "get_network_dim not implemented for AbstractNeuralLyapunovStructure of type " *
            string(typeof(str)) * "."
    )
end

"""
    get_control_structure(str::AbstractNeuralLyapunovStructure{true})

Return the control structure specified by `str`.

# Arguments

- `str`: an `AbstractNeuralLyapunovStructure{true}` instance.

# Returns

A callable `control_structure(phi_c, state, fixed_point)` that transforms the control-output
portion of the neural network into the input expected by the dynamics.

# Extension Rules

This method is required only for `AbstractNeuralLyapunovStructure{true}`. It must preserve
the callable signature so that [`add_policy_search`](@ref) and [`get_policy`](@ref) can use
the result without knowing the concrete subtype.
"""
function get_control_structure(str::AbstractNeuralLyapunovStructure{nc}) where {nc}
    return if nc
        error(
            "control_structure not implemented for AbstractNeuralLyapunovStructure of " *
                "type $(typeof(str))."
        )
    else
        error("control_structure not defined for AbstractNeuralLyapunovStructure{false}.")
    end
end

"""
    get_control_dim(str::AbstractNeuralLyapunovStructure{true})

Return the control dimension specified by `str`.

# Arguments

- `str`: an `AbstractNeuralLyapunovStructure{true}` instance.

# Returns

A positive `Integer` equal to the number of neural-network outputs passed through
`get_control_structure`.
"""
function get_control_dim(str::AbstractNeuralLyapunovStructure{nc}) where {nc}
    return if nc
        error(
            "control_dim not implemented for AbstractNeuralLyapunovStructure of type " *
                string(typeof(str))
        )
    else
        error("control_dim not defined for AbstractNeuralLyapunovStructure{false}.")
    end
end

"""
    neural_controller(str::AbstractNeuralLyapunovStructure)

Return `true` if `str` specifies a neural controller (i.e., if `str` is a subtype of
`AbstractNeuralLyapunovStructure{true}`) and `false` otherwise.

# Arguments

- `str`: the neural Lyapunov structure to classify.

# Returns

`true` exactly for structures parameterized as `AbstractNeuralLyapunovStructure{true}` and
`false` for structures parameterized as `AbstractNeuralLyapunovStructure{false}`.
"""
neural_controller(::AbstractNeuralLyapunovStructure{nc}) where {nc} = nc

"""
    check_nonnegativity(cond::AbstractLyapunovMinimizationCondition)

Return `true` if `cond` specifies training to meet the Lyapunov minimization condition, and
`false` if `cond` specifies no training to meet this condition.

# Arguments

- `cond`: the minimization condition being queried.

# Returns

A `Bool`. When `true`, the generic PDESystem construction consumes the residual returned by
[`get_minimization_condition`](@ref).
"""
function check_nonnegativity(cond::AbstractLyapunovMinimizationCondition)::Bool
    error(
        "check_nonnegativity not implemented for AbstractLyapunovMinimizationCondition " *
            "of type $(typeof(cond))."
    )
end

"""
    check_minimal_fixed_point(cond::AbstractLyapunovMinimizationCondition)

Return `true` if `cond` specifies training for the Lyapunov function to equal zero at the
fixed point, and `false` if `cond` specifies no training to meet this condition.

# Arguments

- `cond`: the minimization condition being queried.

# Returns

A `Bool` controlling whether the generic PDESystem construction adds the equation
`V(fixed_point) = 0`.
"""
function check_minimal_fixed_point(cond::AbstractLyapunovMinimizationCondition)::Bool
    error(
        "check_minimal_fixed_point not implemented for " *
            "AbstractLyapunovMinimizationCondition of type $(typeof(cond))."
    )
end

"""
    get_minimization_condition(cond::AbstractLyapunovMinimizationCondition)

Return a function of ``V``, ``x``, and ``x_0`` that equals zero when the Lyapunov
minimization condition is met for the Lyapunov candidate function ``V`` at the point ``x``,
and is greater than zero if it's violated.

Note that the first input, ``V``, is a function, so the minimization condition can depend on
the value of the candidate Lyapunov function at multiple points.

If the returned function returns a vector, all elements of the vector must be zero for the
condition to be considered met.
[`NeuralLyapunovPDESystem`](@ref) will create one equation per element of the vector.

# Arguments

- `cond`: the minimization condition being queried.

# Returns

Either `nothing` when no nonnegativity equation is requested, or a callable
`(V, state, fixed_point) -> residual` returning a scalar or vector residual.
"""
function get_minimization_condition(cond::AbstractLyapunovMinimizationCondition)
    error(
        "get_minimization_condition not implemented for " *
            "AbstractLyapunovMinimizationCondition of type $(typeof(cond))."
    )
end

function Base.show(io::IO, cond::AbstractLyapunovMinimizationCondition)
    println(io, "AbstractLyapunovMinimizationCondition")
    if check_nonnegativity(cond)
        @variables x x_0 V(..)
        approx_zero = string(get_minimization_condition(cond)(V, x, x_0))
        println(io, "    Trains for $approx_zero ≈ 0")
    else
        println(io, "    Does not train for nonnegativity of V(x)")
    end

    if check_minimal_fixed_point(cond)
        print(io, "    Trains for V(x_0) = 0")
    else
        print(io, "    Does not train for V(x_0) = 0")
    end
    return
end

"""
    check_decrease(cond::AbstractLyapunovDecreaseCondition)

Return `true` if `cond` specifies training to meet the Lyapunov decrease condition, and
`false` if `cond` specifies no training to meet this condition.

# Arguments

- `cond`: the decrease condition being queried.

# Returns

A `Bool`. When `true`, the generic PDESystem construction consumes the residual returned by
[`get_decrease_condition`](@ref).
"""
function check_decrease(cond::AbstractLyapunovDecreaseCondition)::Bool
    error(
        "check_decrease not implemented for AbstractLyapunovDecreaseCondition of type " *
            string(typeof(cond)) * "."
    )
end

"""
    get_decrease_condition(cond::AbstractLyapunovDecreaseCondition)

Return a function of ``V``, ``V̇``, ``x``, and ``x_0`` that returns zero when the Lyapunov
decrease condition is met and a value greater than zero when it is violated.

Note that the first two inputs, ``V`` and ``V̇``, are functions, so the decrease condition
can depend on the value of these functions at multiple points.

If the returned function returns a vector, all elements of the vector must be zero for the
condition to be considered met.
[`NeuralLyapunovPDESystem`](@ref) will create one equation per element of the vector.

# Arguments

- `cond`: the decrease condition being queried.

# Returns

Either `nothing` when no decrease equation is requested, or a callable
`(V, V̇, state, fixed_point) -> residual` returning a scalar or vector residual.
"""
function get_decrease_condition(cond::AbstractLyapunovDecreaseCondition)
    error(
        "get_decrease_condition not implemented for AbstractLyapunovDecreaseCondition " *
            "of type $(typeof(cond))."
    )
end

function Base.show(io::IO, cond::AbstractLyapunovDecreaseCondition)
    println(io, "AbstractLyapunovDecreaseCondition")
    if check_decrease(cond)
        @variables x x_0 V(..) V̇(..)
        approx_zero = string(get_decrease_condition(cond)(V, V̇, x, x_0))
        println(io, "    Trains for $approx_zero ≈ 0")
    else
        print(io, "    Does not train for decrease of V along trajectories")
    end
    return
end
