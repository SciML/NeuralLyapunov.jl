"""
    NoAdditionalStructure()

Create a [`NeuralLyapunovStructure`](@ref) where the Lyapunov function is the neural network
evaluated at the state. This does impose any additional structure to enforce any Lyapunov
conditions.

Corresponds to ``V(x) = ϕ(x)``, where ``ϕ`` is the neural network.

Dynamics are assumed to be in `f(state, p, t)` form, as in an `ODEFunction`. For
`f(state, input, p, t)`, consider using [`add_policy_search`](@ref).

# Example
```jldoctest
julia> NoAdditionalStructure()
NeuralLyapunovStructure
    Network dimension: 1
    V(x) = φ(x)
    V̇(x) = ∇φ(x)*f(x, p, t)
    f_call(x) = f(x, p, t)
```
"""
function NoAdditionalStructure()::NeuralLyapunovStructure
    return NeuralLyapunovStructure(
        (net, x, x0) -> net(x),
        (net, grad_net, f, x, p, t, x0) -> grad_net(x) ⋅ f(x, p, t),
        (f, net, x, p, t) -> f(x, p, t),
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

# Example
```jldoctest
julia> NonnegativeStructure(3; δ = 0.1)
NeuralLyapunovStructure
    Network dimension: 3
    V(x) = 0.1log(1.0 + (x - x_0)²) + ||φ(x)||²
    V̇(x) = 2(φ(x))⋅(f(x, p, t)*Jφ(x)) + (0.2(x - x_0)*f(x, p, t)) / (1.0 + (x - x_0)²)
    f_call(x) = f(x, p, t)
```

See also: [`DontCheckNonnegativity`](@ref)
"""
function NonnegativeStructure(
        network_dim::Integer;
        δ::Real = 0.0,
        pos_def = (x, x0) -> log(1.0 + (x - x0) ⋅ (x - x0)),
        grad_pos_def = nothing,
        grad = ForwardDiff.gradient
    )::NeuralLyapunovStructure
    return if δ == 0.0
        NeuralLyapunovStructure(
            (net, x, x0) -> net(x) ⋅ net(x),
            (net, J_net, f, x, p, t, x0) -> 2 * dot(net(x), J_net(x), f(x, p, t)),
            (f, net, x, p, t) -> f(x, p, t),
            network_dim
        )
    else
        _grad(f, x::AbstractArray{T}) where {T <: Num} = Symbolics.gradient(f(x), x)
        _grad(f, x::T) where {T <: Num} = Symbolics.derivative(f(x), x)
        _grad(f, x) = grad(f, x)
        grad_pos_def = if isnothing(grad_pos_def)
            let __grad = _grad
                (x, x0) -> __grad(Base.Fix2(pos_def, x0), x)
            end
        else
            grad_pos_def
        end
        NeuralLyapunovStructure(
            (net, x, x0) -> net(x) ⋅ net(x) + δ * pos_def(x, x0),
            function (net, J_net, f, x, p, t, x0)
                return 2 * dot(net(x), J_net(x), f(x, p, t)) + δ * grad_pos_def(x, x0) ⋅ f(x, p, t)
            end,
            (f, net, x, p, t) -> f(x, p, t),
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

# Example
```jldoctest
julia> PositiveSemiDefiniteStructure(1)
NeuralLyapunovStructure
    Network dimension: 1
    V(x) = log(1.0 + (x - x_0)^2)*(1 + φ(x)^2)
    V̇(x) = (2(x - x_0)*(1 + φ(x)^2)*f(x, p, t)) / (1.0 + (x - x_0)^2) + 2log(1.0 + (x - x_0)^2)*φ(x)*Differential(x, 1)(φ(x))*f(x, p, t)
    f_call(x) = f(x, p, t)
```

See also: [`DontCheckNonnegativity`](@ref)
"""
function PositiveSemiDefiniteStructure(
        network_dim::Integer;
        pos_def = (x, x0) -> log(1.0 + (x - x0) ⋅ (x - x0)),
        non_neg = (net, x, x0) -> 1 + net(x) ⋅ net(x),
        grad_pos_def = nothing,
        grad_non_neg = nothing,
        grad = ForwardDiff.gradient
    )::NeuralLyapunovStructure
    _grad(f, x::AbstractArray{T}) where {T <: Num} = Symbolics.gradient(f(x), x)
    _grad(f, x::T) where {T <: Num} = Symbolics.derivative(f(x), x)
    _grad(f, x) = grad(f, x)
    grad_pos_def = if isnothing(grad_pos_def)
        let __grad = _grad
            (x, x0) -> __grad(Base.Fix2(pos_def, x0), x)
        end
    else
        grad_pos_def
    end
    grad_non_neg = if isnothing(grad_non_neg)
        let __grad = _grad
            (net, J_net, x, x0) -> __grad((_x) -> non_neg(net, _x, x0), x)
        end
    else
        grad_non_neg
    end
    return NeuralLyapunovStructure(
        (net, x, x0) -> pos_def(x, x0) * non_neg(net, x, x0),
        function (net, J_net, f, x, p, t, x0)
            ẋ = f(x, p, t)
            d_pos_def = ẋ ⋅ grad_pos_def(x, x0)
            d_non_neg = ẋ ⋅ grad_non_neg(net, J_net, x, x0)
            return d_pos_def * non_neg(net, x, x0) + pos_def(x, x0) * d_non_neg
        end,
        (f, net, x, p, t) -> f(x, p, t),
        network_dim
    )
end
