"""
    NeuralLyapunovStructure

Specifies the structure of the neural Lyapunov function and its derivative.

Allows the user to define the Lyapunov in terms of the neural network to structurally
enforce Lyapunov conditions.
`network_dim` is the dimension of the output of the neural network.
`V(phi::Function, state, fixed_point)` takes in the neural network, the state, and the fixed
point, and outputs the value of the Lyapunov function at `state`.
`V̇(phi::Function, J_phi::Function, f::Function, state, params, t, fixed_point)` takes in the
neural network, jacobian of the neural network, dynamics, state, parameters and time (for
calling the dynamics, when relevant), and fixed point, and outputs the time derivative of
the Lyapunov function at `state`.
"""
struct NeuralLyapunovStructure
    V::Function
    ∇V::Function
    V̇::Function
    network_dim::Integer
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
`check_decrease`, `check_stationary_fixed_point`, and `get_decrease_condition` functions.
"""
abstract type AbstractLyapunovDecreaseCondition end

"""
    NeuralLyapunovSpecification

Specifies a neural Lyapunov problem
"""
struct NeuralLyapunovSpecification
    structure::NeuralLyapunovStructure
    minimzation_condition::AbstractLyapunovMinimizationCondition
    decrease_condition::AbstractLyapunovDecreaseCondition
end

"""
check_nonnegativity(cond::AbstractLyapunovMinimizationCondition)

`true` if `cond` specifies training to meet the Lyapunov minimization condition, `false` if
`cond` specifies no training to meet this condition.
"""
function check_nonnegativity(cond::AbstractLyapunovMinimizationCondition)::Bool
    error("check_nonnegativity not implemented for AbstractLyapunovMinimizationCondition " *
          "of type $(typeof(cond))")
end

"""
    check_minimal_fixed_point(cond::AbstractLyapunovMinimizationCondition)

`true` if `cond` specifies training for the Lyapunov function to equal zero at the
fixed point, `false` if `cond` specifies no training to meet this condition.
"""
function check_minimal_fixed_point(cond::AbstractLyapunovMinimizationCondition)::Bool
    error("check_minimal_fixed_point not implemented for " *
          "AbstractLyapunovMinimizationCondition of type $(typeof(cond))")
end

"""
    get_minimization_condition(cond::AbstractLyapunovMinimizationCondition)

Returns a function of `V`, `state`, and `fixed_point` that equals zero when the
Lyapunov minimization condition is met and greater than zero when it's violated.
"""
function get_minimization_condition(cond::AbstractLyapunovMinimizationCondition)
    error("get_condition not implemented for AbstractLyapunovMinimizationCondition of " *
          "type $(typeof(cond))")
end

"""
    check_decrease(cond::AbstractLyapunovDecreaseCondition)

`true` if `cond` specifies training to meet the Lyapunov decrease condition, `false`
if `cond` specifies no training to meet this condition.
"""
function check_decrease(cond::AbstractLyapunovDecreaseCondition)::Bool
    error("check_decrease not implemented for AbstractLyapunovDecreaseCondition of type " *
          string(typeof(cond)))
end

"""
    check_stationary_fixed_point(cond::AbstractLyapunovDecreaseCondition)

`true` if `cond` specifies training for the Lyapunov function not to change at the
fixed point, `false` if `cond` specifies no training to meet this condition.
"""
function check_stationary_fixed_point(cond::AbstractLyapunovDecreaseCondition)::Bool
    error("check_fixed_point not implemented for AbstractLyapunovDecreaseCondition of " *
          "type $(typeof(cond))")
end

"""
    get_decrease_condition(cond::AbstractLyapunovDecreaseCondition)

Returns a function of `V`, `dVdt`, `state`, and `fixed_point` that is equal to zero
when the Lyapunov decrease condition is met and greater than zero when it is violated.
"""
function get_decrease_condition(cond::AbstractLyapunovDecreaseCondition)
    error("get_condition not implemented for AbstractLyapunovDecreaseCondition of type " *
          string((typeof(cond))))
end
