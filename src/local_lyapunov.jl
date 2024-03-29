"""
    get_local_lyapunov(dynamics, state_dim, optimizer_factory[, jac]; fixed_point, p)

Use semidefinite programming to derive a quadratic Lyapunov function for the linearization
of `dynamics` around `fixed_point`. Return `(V, dV/dt)`.

To solve the semidefinite program, `JuMP.Model` requires an `optimizer_factory` capable of
semidefinite programming (SDP). See the
[JuMP documentation](https://jump.dev/JuMP.jl/stable/installation/#Supported-solvers) for
examples.

If `jac` is not supplied, the Jacobian of the `dynamics(x, p, t)` with respect to `x` is
calculated using `ForwardDiff`. Otherwise, `jac` is expected to be either a function or an
`AbstractMatrix`. If `jac isa Function`, it should take in the state and parameters and
output the Jacobian of `dynamics` with respect to the state `x`. If `jac isa
AbstractMatrix`, it should be the value of the Jacobian at `fixed_point`.

If `fixed_point` is not specified, it defaults to the origin, i.e., `zeros(state_dim)`.
Parameters `p` for the dynamics should be supplied when the dynamics depend on them.
"""
function local_lyapunov(dynamics::Function, state_dim, optimizer_factory,
                        jac::AbstractMatrix{T}; fixed_point = zeros(T, state_dim),
                        p = SciMLBase.NullParameters()) where T <: Number

    model = JuMP.Model(optimizer_factory)
    JuMP.set_silent(model)

    # If jac is a matrix A, then the linearization is ẋ = A (x - x0), where
    # x is the state and x0 is the fixed point.
    # A quadratic Lyapunov function is V(x) = (x - x0) ⋅ (P * (x - x0)) for some positive
    # definite matrix P, where V̇(x) = -(x - x0) ⋅ (Q * (x - x0)) for some positive definite
    # matrix Q
    JuMP.@variable(model, P[1:state_dim, 1:state_dim], PSD)
    JuMP.@variable(model, Q[1:state_dim, 1:state_dim], PSD)

    # V̇(x) = (x - x0) ⋅ ( (P A + A^T P) * (x - x0)), so we require P A + A^T P = -Q
    JuMP.@constraint(model, P * jac + transpose(jac) * P.==-Q)

    # Solve the semidefinite program and get the value of P
    JuMP.optimize!(model)
    Psol = JuMP.value.(P)

    # Numerical form of Lyapunov function
    V(state::AbstractVector) = dot(state - fixed_point, Psol, state - fixed_point)
    V(states::AbstractMatrix) = mapslices(V, states, dims = [1])

    # Numerical gradient of Lyapunov function
#    ∇V(state::AbstractVector) = 2 * ( Psol * (state - fixed_point) )
#    ∇V(states::AbstractMatrix) = mapslices(∇V, states, dims = [1])

    # Numerical time derivative of Lyapunov function
    V̇(state::AbstractVector) = 2 * dot(dynamics(state, p, 0.0), Psol, state - fixed_point)
    V̇(states::AbstractMatrix) = mapslices(V̇, states, dims = [1])

    return V, V̇
end

function local_lyapunov(dynamics::Function, state_dim, optimizer_factory, jac::Function;
                        fixed_point = zeros(state_dim), p = SciMLBase.NullParameters())

    A::AbstractMatrix = jac(fixed_point, p)
    return local_lyapunov(
        dynamics,
        state_dim,
        optimizer_factory,
        A;
        fixed_point = fixed_point,
        p = p
    )
end

function local_lyapunov(dynamics::Function, state_dim, optimizer_factory;
                        fixed_point = zeros(state_dim), p = SciMLBase.NullParameters())

    A::AbstractMatrix = ForwardDiff.jacobian(x -> dynamics(x, p, 0.0), fixed_point)
    return local_lyapunov(
        dynamics,
        state_dim,
        optimizer_factory,
        A;
        fixed_point = fixed_point,
        p = p
    )
end
