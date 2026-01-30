"""
    NeuralLyapunovStructure(V, V̇, f_call, network_dim)

Specifies the structure of the neural Lyapunov function and its derivative.

Allows the user to define the Lyapunov in terms of the neural network, potentially
structurally enforcing some Lyapunov conditions.

# Fields
  - `V(phi, state, fixed_point)`: outputs the value of the Lyapunov function at `state`.
  - `V̇(phi, J_phi, dynamics, state, params, t, fixed_point)`: outputs the time derivative of
    the Lyapunov function at `state`.
  - `f_call(dynamics, phi, state, params, t)`: outputs the derivative of the state; this is
    useful for making closed-loop dynamics which depend on the neural network, such as in
    the policy search case.
  - `network_dim`: the dimension of the output of the neural network.

`phi` and `J_phi` above are both functions of `state` alone.
"""
struct NeuralLyapunovStructure{TV, TDV, F, D <: Integer}
    V::TV
    V̇::TDV
    f_call::F
    network_dim::D
end

function Base.show(io::IO, s::NeuralLyapunovStructure)
    n = s.network_dim
    if n > 1
        @variables φ(..)[1:n] Jφ(..)[1:n] f(..) x x_0 p t
        println(io, "NeuralLyapunovStructure")
        println(io, "    Network dimension: ", n)
        try
            println(io, "    V(x) = ", s.V(φ, x, x_0))
        catch e
            println(io, "    V(x) = <could not display: $(e)>")
        end
        try
            V̇ = string(s.V̇(φ, Jφ, f, x, p, t, x_0))
            # Regex to simplify broadcasting notation for better readability
            V̇ = replace(V̇, r"broadcast\(\*,\s*(.+?),\s*Ref\(((?:[^()]|\((?:[^()]|\([^)]*\))*\))*)\)\)" => s"\1 * \2")
            println(io, "    V̇(x) = ", V̇)
        catch e
            println(io, "    V̇(x) = <could not display: $(e)>")
        end
        try
            print(io, "    f_call(x) = ", s.f_call(f, φ, x, p, t))
        catch e
            println(io, "    f_call(x) = <could not display: $(e)>")
        end
    else
        @variables φ(..) ∇φ(..) f(..) x x_0 p t
        println(io, "NeuralLyapunovStructure")
        println(io, "    Network dimension: ", n)
        try
            println(io, "    V(x) = ", s.V(φ, x, x_0))
        catch e
            println(io, "    V(x) = <could not display: $(e)>")
        end
        try
            println(io, "    V̇(x) = ", s.V̇(φ, ∇φ, f, x, p, t, x_0))
        catch e
            println(io, "    V̇(x) = <could not display: $(e)>")
        end
        try
            print(io, "    f_call(x) = ", s.f_call(f, φ, x, p, t))
        catch e
            println(io, "    f_call(x) = <could not display: $(e)>")
        end
    end
    return
end

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
```jldoctest
julia> NeuralLyapunovSpecification(NonnegativeStructure(1), PositiveSemiDefinite(), StabilityISL())
NeuralLyapunovSpecification
    Structure:
        NeuralLyapunovStructure
            Network dimension: 1
            V(x) = φ(x)^2
            V̇(x) = 2φ(x)*f(x, p, t)*∇φ(x)
            f_call(x) = f(x, p, t)
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
    structure::NeuralLyapunovStructure
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
    print(io, replace(string(spec.decrease_condition), r"^(?=.)"m => "        "))
    return
end

"""
    check_nonnegativity(cond::AbstractLyapunovMinimizationCondition)

Return `true` if `cond` specifies training to meet the Lyapunov minimization condition, and
`false` if `cond` specifies no training to meet this condition.
"""
function check_nonnegativity(cond::AbstractLyapunovMinimizationCondition)::Bool
    error(
        "check_nonnegativity not implemented for AbstractLyapunovMinimizationCondition " *
            "of type $(typeof(cond))"
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
            "AbstractLyapunovMinimizationCondition of type $(typeof(cond))"
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
            "AbstractLyapunovMinimizationCondition of type $(typeof(cond))"
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
            string(typeof(cond))
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
            "of type $(typeof(cond))"
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
