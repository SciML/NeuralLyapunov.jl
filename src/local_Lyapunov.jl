"""
    get_local_Lyapunov(dynamics, state_dim; fixed_point, dynamics_jac)

Uses semidefinite programming to derive a quadratic Lyapunov function for the
linearization of dynamics around fixed_point.
Returns (V, dV/dt, ∇V).

If dynamics_jac is nothing, the Jacobian of the dynamics is calculated using 
ForwardDiff. Other allowable forms are a function which takes in the state and
outputs the jacobian of dynamics or an AbstractMatrix representing the Jacobian
at fixed_point. If fixed_point is not specified, it defaults to the origin.
"""
function local_Lyapunov(dynamics::Function, state_dim; fixed_point = zeros(state_dim), dynamics_jac = nothing)
    # Linearize the dynamics
    A = if isnothing(dynamics_jac)
            ForwardDiff.jacobian(dynamics, fixed_point)
        elseif dynamics_jac isa AbstractMatrix
            dynamics_jac
        elseif dynamics_jac isa Function
            dynamics_jac(fixed_point)
        else
            throw(ErrorException("Unable to Calculate Jacobian from dynamics_jac."))
        end
    
    # Use quadratic semidefinite programming to calculate a Lyapunov function
    # for the linearized system
    model = JuMP.Model(Hypatia.Optimizer)
    JuMP.set_silent(model)
    JuMP.@variable(model, P[1:state_dim, 1:state_dim], PSD)
    JuMP.@variable(model, Q[1:state_dim, 1:state_dim], PSD)
    JuMP.@constraint(model, P * A + transpose(A) * P .== -Q)
    JuMP.optimize!(model)
    Psol = JuMP.value.(P)

    # Numerical form of Lyapunov function
    V(state::AbstractVector) = dot(state - fixed_point, Psol, state - fixed_point)
    V(states::AbstractMatrix) = mapslices(V, states, dims = [1])

    # Numerical gradient of Lyapunov function
    ∇V(state::AbstractVector) = 2 * Psol * (state - fixed_point)
    ∇V(states::AbstractMatrix) = mapslices(∇V, states, dims = [1])

    # Numerical time derivative of Lyapunov function
    V̇(state::AbstractVector) = dynamics(state) ⋅ ∇V(state)
    V̇(states::AbstractMatrix) = reshape(
        map(
            x -> x[1] ⋅ x[2],
            zip(
                eachslice(dynamics(states), dims = 2),
                eachslice(∇V(states), dims = 2),
            ),
        ),
        (1, :),
    )

    return V, V̇, ∇V
end