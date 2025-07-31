"""
    NoAdditionalStructure()

Create a [`NeuralLyapunovStructure`](@ref) where the Lyapunov function is the neural network
evaluated at the state. This does impose any additional structure to enforce any Lyapunov
conditions.

Corresponds to ``V(x) = ϕ(x)``, where ``ϕ`` is the neural network.

Dynamics are assumed to be in `f(state, p, t)` form, as in an `ODEFunction`. For
`f(state, input, p, t)`, consider using [`add_policy_search`](@ref).
"""
function NoAdditionalStructure()::NeuralLyapunovStructure
    NeuralLyapunovStructure(
        (net, state, fixed_point) -> net(state),
        (net, grad_net, f, state, params, t,
            fixed_point) -> grad_net(state) ⋅
                            f(state, params, t),
        (f, net, state, p, t) -> f(state, p, t),
        1
    )
end

"""
    NonnegativeStructure(network_dim; <keyword_arguments>)

Create a [`NeuralLyapunovStructure`](@ref) where the Lyapunov function is the L2 norm of the
neural network output plus a constant δ times a function `pos_def`.

Corresponds to ``V(x) = \\lVert ϕ(x) \\rVert^2 + δ \\, \\texttt{pos\\_def}(x, x_0)``, where
``ϕ`` is the neural network and ``x_0`` is the equilibrium point.

This structure ensures ``V(x) ≥ 0 \\, ∀ x`` when ``δ ≥ 0`` and `pos_def` is always
nonnegative. Further, if ``δ > 0`` and `pos_def` is strictly positive definite around
`fixed_point`, the structure ensures that ``V(x)`` is strictly positive away from
`fixed_point`. In such cases, the minimization condition reduces to ensuring
``V(x_0) = 0``, and so [`DontCheckNonnegativity(true)`](@ref) should be
used.

# Arguments
  - `network_dim`: output dimensionality of the neural network.

# Keyword Arguments
  - `δ`: weight of `pos_def`, as above; defaults to 0.
  - `pos_def(state, fixed_point)`: a function that is positive (semi-)definite in `state`
    around `fixed_point`; defaults to ``\\log(1 + \\lVert x - x_0 \\rVert^2)``.
  - `grad_pos_def(state, fixed_point)`: the gradient of `pos_def` with respect to `state` at
    `state`. If `isnothing(grad_pos_def)` (as is the default), the gradient of `pos_def`
    will be evaluated using `grad`.
  - `grad`: a function for evaluating gradients to be used when `isnothing(grad_pos_def)`;
    defaults to, and expects the same arguments as, `ForwardDiff.gradient`.

Dynamics are assumed to be in `f(state, p, t)` form, as in an `ODEFunction`. For
`f(state, input, p, t)`, consider using [`add_policy_search`](@ref).

See also: [`DontCheckNonnegativity`](@ref)
"""
function NonnegativeStructure(
        network_dim::Integer;
        δ::Real = 0.0,
        pos_def::Function = (
            state, fixed_point) -> log(1.0 +
                                       (state - fixed_point) ⋅
                                       (state - fixed_point)),
        grad_pos_def = nothing,
        grad = ForwardDiff.gradient
)::NeuralLyapunovStructure
    if δ == 0.0
        NeuralLyapunovStructure(
            (net, state, fixed_point) -> net(state) ⋅ net(state),
            (net, J_net, f, state, params, t,
                fixed_point) -> 2 * dot(
                net(state), J_net(state), f(state, params, t)),
            (f, net, state, p, t) -> f(state, p, t),
            network_dim
        )
    else
        grad_pos_def = if isnothing(grad_pos_def)
            (state, fixed_point) -> grad((x) -> pos_def(x, fixed_point), state)
        else
            grad_pos_def
        end
        NeuralLyapunovStructure(
            (net, state,
                fixed_point) -> net(state) ⋅ net(state) +
                                δ * pos_def(state, fixed_point),
            (net,
                J_net,
                f,
                state,
                params,
                t,
                fixed_point) -> 2 * dot(
                net(state),
                J_net(state),
                f(state, params, t)
            ) + δ * grad_pos_def(state, fixed_point) ⋅ f(state, params, t),
            (f, net, state, p, t) -> f(state, p, t),
            network_dim
        )
    end
end

"""
    PositiveSemiDefiniteStructure(network_dim; <keyword_arguments>)

Create a [`NeuralLyapunovStructure`](@ref) where the Lyapunov function is the product of a
positive (semi-)definite function `pos_def` which does not depend on the network and a
nonnegative function `non_neg` which does depend the network.

Corresponds to ``V(x) = \\texttt{pos\\_def}(x, x_0) * \\texttt{non\\_neg}(ϕ, x, x_0)``, where
``ϕ`` is the neural network and ``x_0`` is the equilibrium point.

This structure ensures ``V(x) ≥ 0``. Further, if `pos_def` is strictly positive definite
`fixed_point` and `non_neg` is strictly positive (as is the case for the default values of
`pos_def` and `non_neg`), then this structure ensures ``V(x)`` is strictly positive definite
around `fixed_point`. In such cases, the minimization condition is satisfied structurally,
so [`DontCheckNonnegativity(false)`](@ref) should be used.

# Arguments
  - network_dim: output dimensionality of the neural network.

# Keyword Arguments
  - `pos_def(state, fixed_point)`: a function that is positive (semi-)definite in `state`
    around `fixed_point`; defaults to ``\\log(1 + \\lVert x - x_0 \\rVert^2)``.
  - `non_neg(net, state, fixed_point)`: a nonnegative function of the neural network; note
    that `net` is the neural network ``ϕ``, and `net(state)` is the value of the neural
    network at a point ``ϕ(x)``; defaults to ``1 + \\lVert ϕ(x) \\rVert^2``.
  - `grad_pos_def(state, fixed_point)`: the gradient of `pos_def` with respect to `state` at
    `state`. If `isnothing(grad_pos_def)` (as is the default), the gradient of `pos_def`
    will be evaluated using `grad`.
  - `grad_non_neg(net, J_net, state, fixed_point)`: the gradient of `non_neg` with respect
    to `state` at `state`; `J_net` is a function outputting the Jacobian of `net` at the
    input. If `isnothing(grad_non_neg)` (as is the default), the gradient of `non_neg` will
    be evaluated using `grad`.
  - `grad`: a function for evaluating gradients to be used when `isnothing(grad_pos_def) ||
    isnothing(grad_non_neg)`; defaults to, and expects the same arguments as,
    `ForwardDiff.gradient`.

Dynamics are assumed to be in `f(state, p, t)` form, as in an `ODEFunction`. For
`f(state, input, p, t)`, consider using [`add_policy_search`](@ref).

See also: [`DontCheckNonnegativity`](@ref)
"""
function PositiveSemiDefiniteStructure(
        network_dim::Integer;
        pos_def::Function = (
            state, fixed_point) -> log(1.0 +
                                       (state - fixed_point) ⋅
                                       (state - fixed_point)),
        non_neg::Function = (net, state, fixed_point) -> 1 + net(state) ⋅ net(state),
        grad_pos_def = nothing,
        grad_non_neg = nothing,
        grad = ForwardDiff.gradient
)::NeuralLyapunovStructure
    _grad(f::Function, x::AbstractArray{T}) where {T <: Num} = Symbolics.gradient(f(x), x)
    _grad(f::Function, x) = grad(f, x)
    grad_pos_def = if isnothing(grad_pos_def)
        (state, fixed_point) -> _grad((x) -> pos_def(x, fixed_point), state)
    else
        grad_pos_def
    end
    grad_non_neg = if isnothing(grad_non_neg)
        (net, J_net, state,
            fixed_point) -> _grad(
            (x) -> non_neg(net, x, fixed_point), state)
    else
        grad_non_neg
    end
    NeuralLyapunovStructure(
        (net, state,
            fixed_point) -> pos_def(state, fixed_point) *
                            non_neg(net, state, fixed_point),
        (net,
            J_net,
            f,
            state,
            params,
            t,
            fixed_point) -> (f(state, params, t) ⋅
                             grad_pos_def(
            state, fixed_point)) *
                            non_neg(net, state, fixed_point) +
                            pos_def(state, fixed_point) *
                            (f(state, params, t) ⋅
                             grad_non_neg(
            net, J_net, state, fixed_point)),
        (f, net, state, p, t) -> f(state, p, t),
        network_dim
    )
end
