"""
    AbstractNeuralLyapunovStructure{nc}

Represents the structure of the neural Lyapunov function and its derivative.

All concrete `AbstractNeuralLyapunovStructure` subtypes should define the `get_V`, `get_V̇`,
and `get_network_dim` functions. If `nc` is `true`, the subtype should also define the
`get_control_structure` and `get_control_dim` functions.
"""
abstract type AbstractNeuralLyapunovStructure{nc} end

"""
    AbstractLyapunovMinimizationCondition

Represents the minimization condition in a neural Lyapunov problem

All concrete `AbstractLyapunovMinimizationCondition` subtypes should define the
`check_nonnegativity`, `check_fixed_point`, and `get_minimization_condition` functions.
"""
abstract type AbstractLyapunovMinimizationCondition end

"""
    AbstractLyapunovDecreaseCondition

Represents the decrease condition in a neural Lyapunov problem

All concrete `AbstractLyapunovDecreaseCondition` subtypes should define the
`check_decrease` and `get_decrease_condition` functions.
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
"""
function get_V̇(str::AbstractNeuralLyapunovStructure)
    error(
        "get_V̇ not implemented for AbstractNeuralLyapunovStructure of type " *
            string(typeof(str)) * "."
    )
end

"""
    get_network_dim(str::AbstractNeuralLyapunovStructure)

Return the number of dimensions of the neural network output specified by `spec`.
"""
function get_network_dim(str::AbstractNeuralLyapunovStructure)
    error(
        "get_network_dim not implemented for AbstractNeuralLyapunovStructure of type " *
            string(typeof(str)) * "."
    )
end

"""
    get_control_structure(str::AbstractNeuralLyapunovStructure{true})

Return the control structure specified by `spec`.
"""
function get_control_structure(str::AbstractNeuralLyapunovStructure{nc}) where nc
    if nc
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

Return the control dimension specified by `spec`.
"""
function get_control_dim(str::AbstractNeuralLyapunovStructure{nc}) where nc
    if nc
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
"""
neural_controller(::AbstractNeuralLyapunovStructure{nc}) where nc = nc

"""
    check_nonnegativity(cond::AbstractLyapunovMinimizationCondition)

Return `true` if `cond` specifies training to meet the Lyapunov minimization condition, and
`false` if `cond` specifies no training to meet this condition.
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
