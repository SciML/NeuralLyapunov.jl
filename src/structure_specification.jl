"""
    UnstructuredNeuralLyapunov()

Create a `NeuralLyapunovStructure` where the Lyapunov function is the neural network
evaluated at the state. This does not structurally enforce any Lyapunov conditions.

Dynamics are assumed to be in `f(state, p, t)` form, as in an `ODEFunction`. For
`f(state, input, p, t)`, consider using `add_policy_search`.
"""
function UnstructuredNeuralLyapunov()::NeuralLyapunovStructure
    NeuralLyapunovStructure(
        (net, state, fixed_point) -> net(state),
        (net, grad_net, state, fixed_point) -> grad_net(state),
        (net, grad_net, f, state, params, t, fixed_point) -> grad_net(state) ⋅
                                                             f(state, params, t),
        (f, net, state, p, t) -> f(state, p, t),
        1
    )
end

"""
    NonnegativeNeuralLyapunov(network_dim, δ, pos_def; grad_pos_def, grad)

Create a `NeuralLyapunovStructure` where the Lyapunov function is the L2 norm of the neural
network output plus a constant δ times a function `pos_def`.

The condition that the Lyapunov function must be minimized uniquely at the fixed point can
be represented as `V(fixed_point) = 0`, `V(state) > 0` when `state ≠ fixed_point`. This
structure ensures `V(state) ≥ 0`. Further, if `δ > 0` and
`pos_def(fixed_point, fixed_point) = 0`, but `pos_def(state, fixed_point) > 0` when
`state ≠ fixed_point`, this ensures that `V(state) > 0` when `state != fixed_point`. This
does not enforce `V(fixed_point) = 0`, so that condition must included in the neural
Lyapunov loss function.

`grad_pos_def(state, fixed_point)` should be the gradient of `pos_def` with respect to
`state` at `state`. If `grad_pos_def` is not defined, it is evaluated using `grad`, which
defaults to `ForwardDiff.gradient`.

The neural network output has dimension `network_dim`.

Dynamics are assumed to be in `f(state, p, t)` form, as in an `ODEFunction`. For
`f(state, input, p, t)`, consider using `add_policy_search`.
"""
function NonnegativeNeuralLyapunov(
        network_dim::Integer;
        δ::Real = 0.0,
        pos_def::Function = (state, fixed_point) -> log(1.0 +
                                                        (state - fixed_point) ⋅
                                                        (state - fixed_point)),
        grad_pos_def = nothing,
        grad = ForwardDiff.gradient
)::NeuralLyapunovStructure
    if δ == 0.0
        NeuralLyapunovStructure(
            (net, state, fixed_point) -> net(state) ⋅ net(state),
            (net, J_net, state, fixed_point) -> 2 * transpose(net(state)) * J_net(state),
            (net, J_net, f, state, params, t, fixed_point) -> 2 *
                                                              dot(
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
            (net, state, fixed_point) -> net(state) ⋅ net(state) +
                                         δ * pos_def(state, fixed_point),
            (net, J_net, state, fixed_point) -> 2 * transpose(net(state)) * J_net(state) +
                                                δ * grad_pos_def(state, fixed_point),
            (net, J_net, f, state, params, t, fixed_point) -> 2 * dot(
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
    PositiveSemiDefiniteStructure(network_dim; pos_def, non_neg, grad_pos_def, grad_non_neg, grad)

Create a `NeuralLyapunovStructure` where the Lyapunov function is the product of a positive
(semi-)definite function `pos_def` which does not depend on the network and a nonnegative
function non_neg which does depend the network.

The condition that the Lyapunov function must be minimized uniquely at the fixed point can
be represented as `V(fixed_point) = 0`, `V(state) > 0` when `state ≠ fixed_point`. This
structure ensures `V(state) ≥ 0`. Further, if `pos_def` is `0` only at `fixed_point` (and
positive elsewhere) and if `non_neg` is strictly positive away from `fixed_point` (as is the
case for the default values of `pos_def` and `non_neg`), then this structure ensures
`V(fixed_point) = 0` and `V(state) > 0` when `state ≠ fixed_point`.

`grad_pos_def(state, fixed_point)` should be the gradient of `pos_def` with respect to
`state` at `state`. Similarly, `grad_non_neg(net, J_net, state, fixed_point)` should be the
gradient of `non_neg(net, state, fixed_point)` with respect to `state` at `state`. If
`grad_pos_def` or `grad_non_neg` is not defined, it is evaluated using `grad`, which
defaults to `ForwardDiff.gradient`.

The neural network output has dimension `network_dim`.

Dynamics are assumed to be in `f(state, p, t)` form, as in an `ODEFunction`. For
`f(state, input, p, t)`, consider using `add_policy_search`.
"""
function PositiveSemiDefiniteStructure(
        network_dim::Integer;
        pos_def::Function = (state, fixed_point) -> log(1.0 +
                                                        (state - fixed_point) ⋅
                                                        (state - fixed_point)),
        non_neg::Function = (net, state, fixed_point) -> 1 + net(state) ⋅ net(state),
        grad_pos_def = nothing,
        grad_non_neg = nothing,
        grad = ForwardDiff.gradient
)
    _grad(f::Function, x::AbstractArray{T}) where {T <: Num} = Symbolics.gradient(f(x), x)
    _grad(f::Function, x) = grad(f, x)
    grad_pos_def = if isnothing(grad_pos_def)
        (state, fixed_point) -> _grad((x) -> pos_def(x, fixed_point), state)
    else
        grad_pos_def
    end
    grad_non_neg = if isnothing(grad_non_neg)
        (net, J_net, state, fixed_point) -> _grad(
            (x) -> non_neg(net, x, fixed_point), state)
    else
        grad_non_neg
    end
    NeuralLyapunovStructure(
        (net, state, fixed_point) -> pos_def(state, fixed_point) *
                                     non_neg(net, state, fixed_point),
        (net, J_net, state, fixed_point) -> grad_pos_def(state, fixed_point) *
                                            non_neg(net, state, fixed_point) +
                                            pos_def(state, fixed_point) *
                                            grad_non_neg(net, J_net, state, fixed_point),
        (net, J_net, f, state, params, t, fixed_point) -> (f(state, params, t) ⋅
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
