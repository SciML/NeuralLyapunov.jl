"""
    NeuralLyapunovStructure(V, V̇, network_dim)

Specifies the structure of the neural Lyapunov function and its derivative.

Allows the user to define the Lyapunov in terms of the neural network, potentially
structurally enforcing some Lyapunov conditions.

# Fields
  - `V(phi, state, fixed_point)`: outputs the value of the Lyapunov function at `state`.
  - `V̇(phi, J_phi, state, d_state_dt, fixed_point)`: outputs the time derivative of the
    Lyapunov function at `state`.
  - `network_dim`: the dimension of the output of the neural network.

`phi` and `J_phi` above are both functions of `state` alone.
"""
struct NeuralLyapunovStructure{TV, TDV, D <: Integer} <: AbstractNeuralLyapunovStructure{false}
    V::TV
    V̇::TDV
    network_dim::D
end

function Base.show(io::IO, s::NeuralLyapunovStructure)
    n = s.network_dim
    if n > 1
        @variables φ(..)[1:n] Jφ(..)[1:n, 1:1] x dxdt x_0 p t
        println(io, "NeuralLyapunovStructure")
        println(io, "    Network dimension: ", n)
        try
            V = string(s.V(φ, x, x_0))
            # Regex to simplify broadcasting notation for better readability
            # Replace, e.g., [1:1] with [1]
            V = replace(V, r"\b(\d+):\1\b" => s"\1")
            # Replace LinearAlgebra.dot(A, A) with ||A||^2 for better readability
            dot_re = r"LinearAlgebra\.dot\(\s*((?:[^()]+|\((?1)\))*)\s*,\s*((?:[^()]+|\((?1)\))*)\s*\)"
            V = replace(V, dot_re => s"||\1||²")
            # Replace abs2(A) with ||A||^2 for better readability
            V = replace(V, r"abs2\(\s*((?:[^()]+|\((?1)\))*)\s*\)" => s"||\1||²")
            # Replace LinearAlgebra.dot with ⋅ for better readability
            dot_re = r"LinearAlgebra\.dot\(\s*((?:[^()]+|\((?1)\))*)\s*,\s*((?:[^()]+|\((?2)\))*)\s*\)"
            V = replace(V, dot_re => s"(\1)⋅(\2)")
            # Replace ^2 with ²
            V = replace(V, r"\^2" => "²")
            println(io, "    V(x) = ", V)
        catch e
            println(io, "    V(x) = <could not display: $e>")
        end
        try
            V̇ = string(s.V̇(φ, Jφ, x, dxdt, x_0))
            # Regex to simplify broadcasting notation for better readability
            # Replace, e.g., [1:2, Colon()] with [1:2]
            V̇ = replace(V̇, r",\s*Colon\(\)" => "")
            # Replace, e.g., [1:1] with [1]
            V̇ = replace(V̇, r"\b(\d+):\1\b" => s"\1")
            # Replace abs2(A) with ||A||^2 for better readability
            V̇ = replace(V̇, r"abs2\(\s*((?:[^()]+|\((?1)\))*)\s*\)" => s"||\1||²")
            # Replace LinearAlgebra.dot with ⋅ for better readability
            dot_re = r"LinearAlgebra\.dot\(\s*((?:[^()]+|\((?1)\))*)\s*,\s*((?:[^()]+|\((?2)\))*)\s*\)"
            V̇ = replace(V̇, dot_re => s"(\1)⋅(\2)")
            # Replace ^2 with ²
            V̇ = replace(V̇, r"\^2" => "²")
            # Replace Differential(x, 1)(φ(x)) with Jφ(x) for better readability
            V̇ = replace(V̇, r"Differential\(x, 1\)\(φ\(x\)\)" => "Jφ(x)")
            # Replace dxdt with ẋ for better readability
            V̇ = replace(V̇, r"dxdt" => "ẋ")
            println(io, "    V̇(x) = ", V̇)
        catch e
            println(io, "    V̇(x) = <could not display: $(e)>")
        end
    else
        @variables φ(..) ∇φ(..) x dxdt x_0 p t
        println(io, "NeuralLyapunovStructure")
        println(io, "    Network dimension: ", n)
        try
            V = string(s.V(φ, x, x_0))
            # Replace abs2(A) with ||A||² for better readability
            V = replace(V, r"abs2\(\s*((?:[^()]+|\((?1)\))*)\s*\)" => s"||\1||²")
            # Replace ^2 with ²
            V = replace(V, r"\^2" => "²")
            println(io, "    V(x) = ", V)
        catch e
            println(io, "    V(x) = <could not display: $(e)>")
        end
        try
            V̇ = string(s.V̇(φ, ∇φ, x, dxdt, x_0))
            # Replace abs2(A) with ||A||^2 for better readability
            V̇ = replace(V̇, r"abs2\(\s*((?:[^()]+|\((?1)\))*)\s*\)" => s"||\1||²")
            # Replace ^2 with ²
            V̇ = replace(V̇, r"\^2" => "²")
            # Replace Differential(x, 1)(φ(x)) with ∇φ(x) for better readability
            V̇ = replace(V̇, r"Differential\(x, 1\)\(φ\(x\)\)" => "∇φ(x)")
            # Replace dxdt with ẋ for better readability
            V̇ = replace(V̇, r"dxdt" => "ẋ")
            println(io, "    V̇(x) = ", V̇)
        catch e
            println(io, "    V̇(x) = <could not display: $(e)>")
        end
    end
    return
end

get_V(spec::NeuralLyapunovStructure) = spec.V
get_V̇(spec::NeuralLyapunovStructure) = spec.V̇
get_network_dim(spec::NeuralLyapunovStructure) = spec.network_dim
neural_controller(::NeuralLyapunovStructure) = false

"""
    NoAdditionalStructure()

Create a [`NeuralLyapunovStructure`](@ref) where the Lyapunov function is the neural network
evaluated at the state. This does impose any additional structure to enforce any Lyapunov
conditions.

Corresponds to ``V(x) = ϕ(x)``, where ``ϕ`` is the neural network.

Dynamics are assumed to be in `f(state, p, t)` form, as in an `ODEFunction`. For
`f(state, input, p, t)`, consider using [`add_policy_search`](@ref).

# Example
```jldoctest; filter = [r"ẋ" => "ẋ", r"φ\\(x\\)" => "φ", r"(?m)\\s*V̇\\(x\\)\\s*=\\s*(∇φ\\s*\\*\\s*ẋ|ẋ\\s*\\*\\s*∇φ)\$"]
julia> NoAdditionalStructure()
NeuralLyapunovStructure
    Network dimension: 1
    V(x) = φ(x)
    V̇(x) = ∇φ(x)*ẋ
```
"""
function NoAdditionalStructure()::NeuralLyapunovStructure
    return NeuralLyapunovStructure(
        (net, x, x0) -> net(x),
        (net, grad_net, x, ẋ, x0) -> grad_net(x) ⋅ ẋ,
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
```jldoctest; filter = [r"\\(?x - x_0\\)?" => "Δ", r"ẋ" => "ẋ", r"\\s*\\*\\s*" => "", r"φ\\(x\\)|\\(φ\\(x\\)\\)" => "φ", r"\\|\\|φ\\|\\|²" => "φ²", r"0.1\\s*log\\((1\\s*\\+\\s*Δ²|Δ²\\s*\\+\\s*1)\\)" => "A", r"(?m)\\s*V\\(x\\)\\s*=\\s*(A\\s*\\+\\s*φ²|φ²\\s*\\+\\s*A)\$", r"(ẋJφ|Jφẋ|\\(ẋJφ\\)|\\(Jφẋ\\))" => "φ̇", r"2(φ⋅φ̇|φ̇⋅φ)" => "B", r"\\(?(1\\s*\\+\\s*Δ²|Δ²\\s*\\+\\s*1)\\)?"=> "C", r"\\(?0.2(Δẋ|ẋΔ)\\)?" => "D", r"D\\s*/\\s*C" => "E", r"(?m)\\s*V̇\\(x\\)\\s*=\\s*(B\\s*\\+\\s*E|E\\s*\\+\\s*B)\$"]
julia> NonnegativeStructure(3; δ = 0.1)
NeuralLyapunovStructure
    Network dimension: 3
    V(x) = 0.1log(1 + (x - x_0)²) + ||φ(x)||²
    V̇(x) = 2(φ(x))⋅(ẋ*Jφ(x)) + (0.2(x - x_0)*ẋ) / (1 + (x - x_0)²)
```

See also: [`DontCheckNonnegativity`](@ref)
"""
function NonnegativeStructure(
        network_dim::Integer;
        δ::Real = 0,
        pos_def = (x, x0) -> log(1 + (x - x0) ⋅ (x - x0)),
        grad_pos_def = nothing,
        grad = ForwardDiff.gradient
    )::NeuralLyapunovStructure
    return if δ == 0
        NeuralLyapunovStructure(
            (net, x, x0) -> net(x) ⋅ net(x),
            (net, J_net, x, ẋ, x0) -> 2 * dot(net(x), J_net(x), ẋ),
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
            function (net, J_net, x, ẋ, x0)
                return 2 * dot(net(x), J_net(x), ẋ) + δ * grad_pos_def(x, x0) ⋅ ẋ
            end,
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
```jldoctest; filter = [r"\\(?2x - 2x_0\\)?" => "2(x - x_0)", r"\\(?x - x_0\\)?" => "Δ", r"\\|\\|Δ\\|\\|²" => "Δ²", r"φ\\(x\\)" => "φ", r"\\(?1\\s*\\+\\s*φ²\\)?" => "Y", r"\\s*\\*\\s*" => "", r"(?m)\\s*V\\(x\\)\\s*=\\s*(?:Δ²Y|YΔ²)\$", r"ẋ" => "ẋ", r"(2ΔẋY|2ΔYẋ|ẋ2ΔY|ẋY2Δ|Yẋ2Δ|Y2Δẋ)" => "A", r"(2|∇φ|ẋ|Δ²|φ){5}" => "B", r"(?m)\\s*V̇\\(x\\)\\s*=\\s*(A|B)\\s*\\+\\s*(A|B)\$"]
julia> PositiveSemiDefiniteStructure(1; pos_def = (x, x0) -> sum(abs2, x - x0))
NeuralLyapunovStructure
    Network dimension: 1
    V(x) = (1 + φ(x)²)*||x - x_0||²
    V̇(x) = (2x - 2x_0)*ẋ*(1 + φ(x)²) + 2∇φ(x)*ẋ*||x - x_0||²*φ(x)
```

See also: [`DontCheckNonnegativity`](@ref)
"""
function PositiveSemiDefiniteStructure(
        network_dim::Integer;
        pos_def = (x, x0) -> log(1 + (x - x0) ⋅ (x - x0)),
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
        function (net, J_net, x, ẋ, x0)
            d_pos_def = ẋ ⋅ grad_pos_def(x, x0)
            d_non_neg = ẋ ⋅ grad_non_neg(net, J_net, x, x0)
            return d_pos_def * non_neg(net, x, x0) + pos_def(x, x0) * d_non_neg
        end,
        network_dim
    )
end
