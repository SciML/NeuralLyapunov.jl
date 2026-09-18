"""
    get_numerical_lyapunov_function(phi, θ, structure, dynamics, fixed_point;
                                    <keyword_arguments>)

Combine Lyapunov function structure, dynamics, and neural network weights to generate Julia
functions representing the Lyapunov function and its time derivative: ``V(x), V̇(x)``.

These functions can operate on a state vector or columnwise on a matrix of state vectors.

The parameters `θ` of the neural network that are used in the returned functions remain on
the same device (e.g., CPU or GPU) as they are passed in. If `θ` is on the GPU, users must
ensure that `dynamics` can operate on GPU arrays (e.g., be careful about scalar indexing).

# Positional Arguments
  - `phi`: the neural network, represented as a `NeuralPDE.Phi` object if the neural network
    has a single output, or an `AbsractVector{<:Phi}` with one entry per neural network
    output.
  - `θ`: the parameters of the neural network; If the neural network has multiple outputs,
    `θ[:φ1]` should be the parameters of the first neural network output, `θ[:φ2]` the
    parameters of the second (if there are multiple), and so on. If the neural network has a
    single output, `θ` should be the parameters of the network.
  - `structure`: a [`NeuralLyapunovStructure`](@ref) representing the structure of the
    neural Lyapunov function.
  - `dynamics`: the system dynamics, as a function `ẋ = f(x[, u], p, t)`.

# Keyword Arguments
  - `fixed_point`: the equilibrium point being analyzed by the Lyapunov function.
  - `p`: parameters to be passed into `dynamics`; defaults to `SciMLBase.NullParameters()`.
  - `use_V̇_structure`: when `true`, ``V̇(x)`` is calculated using `structure.V̇`; when `
    false`, ``V̇(x)`` is calculated using `deriv` as ``\\frac{∂}{∂t} V(x + t f(x))`` at
    ``t = 0``; defaults to `false`, as it is more efficient in many cases.
  - `deriv`: a function for calculating derivatives; defaults to (and expects same arguments
    as) `DifferentiationInterface.derivative` with an `AutoForwardDiff` backend; only used
    when `use_V̇_structure` is `false`.
  - `jac`: a function for calculating Jacobians; defaults to (and expects same arguments as)
    `DifferentiationInterface.jacobian` with an `AutoForwardDiff` backend; only used when
    `use_V̇_structure` is `true`.
  - `J_net`: the Jacobian of the neural network, specified as a function
    `J_net(phi, θ, state)`; if `isnothing(J_net)` (as is the default), `J_net` will be
    calculated using `jac`; only used when `use_V̇_structure` is `true`.
"""
function get_numerical_lyapunov_function(
        phi::Union{Phi, AbstractVector{<:Phi}},
        θ,
        structure::AbstractNeuralLyapunovStructure{nc},
        dynamics;
        fixed_point = nothing,
        p = SciMLBase.NullParameters(),
        use_V̇_structure::Bool = false,
        deriv = _forward_derivative,
        jac = _forward_jacobian,
        J_net = nothing
    ) where {nc}
    V = NumericLyapunovFunction(phi, structure, θ; fixed_point)

    if use_V̇_structure
        if nc
            V̇ = StructuredNumericLyapunovControlFunction(
                phi,
                structure,
                θ,
                dynamics;
                p,
                jac,
                J_phi = J_net,
                fixed_point
            )
        else
            V̇ = StructuredNumericLyapunovDecreaseFunction(
                phi,
                structure,
                θ,
                dynamics;
                p,
                jac,
                J_phi = J_net,
                fixed_point
            )
        end
    else
        if nc
            u_dim = get_control_dim(structure)
            φ_dim = get_network_dim(structure) - u_dim
            control_smodel = phi_to_net(phi, θ; idx = (φ_dim + 1):(φ_dim + u_dim))
            control_structure = get_control_structure(structure)

            V̇ = ADNumericLyapunovControlFunction(
                V,
                dynamics,
                control_smodel,
                control_structure;
                p,
                fixed_point,
                deriv
            )
        else
            V̇ = ADNumericLyapunovDecreaseFunction(V, dynamics; p, deriv)
        end

    end

    return V, V̇
end

"""
    phi_to_net(phi, θ[; idx])

Return the network as a function of state alone.

# Arguments
  - `phi`: the neural network, represented as `phi(x, θ)` if the neural network has a single
    output, or an `AbstractVector` of the same with one entry per neural network output;
    typically this is the `phi` field of the output of `NeuralPDE.PhysicsInformedNN`.
    When `phi isa NeuralPDE.Phi` (or an `AbstractVector` thereof), the returned function is
    a `StatefulLuxLayer` that can be called on a state vector.
  - `θ`: the parameters of the neural network; If the neural network has multiple outputs,
    `θ[:φ1]` should be the parameters of the first neural network output, `θ[:φ2]` the
    parameters of the second (if there are multiple), and so on. If the neural network has a
    single output, `θ` should be the parameters of the network.
  - `idx`: the neural network outputs to include in the returned function; defaults to all
    and only applicable when `phi isa AbstractVector`. For each `i` in `idx`, `:φi` must be
    a key in `θ`.
"""
phi_to_net(phi, θ) = Base.Fix2(phi, θ)

function phi_to_net(phi::AbstractVector, θ; idx = eachindex(phi))
    let _θ = θ, φ = phi, _idx = idx
        return function (x)
            return reduce(
                vcat,
                Array(φ[i](x, _θ[Symbol(:φ, i)])) for i in _idx
            )
        end
    end
end

function phi_to_net(phi::Phi, θ)
    model = phi.smodel.model
    st = phi.smodel.st
    return StatefulLuxLayer{true}(model, θ, st)
end

function phi_to_net(phi::AbstractVector{<:Phi}, θ; idx = eachindex(phi))
    models = NamedTuple(map(((i, φ),) -> Symbol(:φ, i) => φ.smodel.model, zip(idx, phi[idx])))
    model = Parallel(vcat; models...)

    θ = θ[Tuple(Symbol(:φ, i) for i in idx)]
    st = NamedTuple(map(((i, φ),) -> Symbol(:φ, i) => φ.smodel.st, zip(idx, phi[idx])))

    return StatefulLuxLayer{true}(model, θ, st)
end

struct NumericLyapunovFunction{
        S <: StatefulLuxLayer, DΘ, V, X0 <: Union{Nothing, AbstractVector{<:Real}},
    }
    smodel::S
    devθ::DΘ
    V_structure::V
    fixed_point::X0
end

function NumericLyapunovFunction(
        phi::Union{Phi, AbstractVector{<:Phi}},
        structure::AbstractNeuralLyapunovStructure,
        θ;
        fixed_point = nothing
    )
    if neural_controller(structure)
        phi_dim = get_network_dim(structure) - get_control_dim(structure)
        smodel = phi_to_net(phi, θ; idx = 1:phi_dim)
    else
        smodel = phi_to_net(phi, θ)
    end

    V_structure = get_V(structure)

    return NumericLyapunovFunction(smodel, safe_get_device(θ), V_structure, fixed_point)
end

function (V::NumericLyapunovFunction)(x::AbstractVector)
    x0 = isnothing(V.fixed_point) ? zero(x) : V.fixed_point
    return V.V_structure(V.smodel, V.devθ(x), V.devθ(x0))
end

(V::NumericLyapunovFunction)(x::AbstractMatrix) = mapslices(V, x, dims = [1])

struct StructuredNumericLyapunovControlFunction{S <: StatefulLuxLayer, JS, DΘ, P, X0 <: Union{Nothing, AbstractVector{<:Real}}, US, U, DV}
    smodel::S
    smodel_jac::JS
    devθ::DΘ
    dynamics::ODEInputFunction
    p::P
    fixed_point::X0
    control_smodel::US
    control_structure::U
    V̇_structure::DV
end

function StructuredNumericLyapunovControlFunction(
        phi::AbstractVector{<:Phi},
        structure::AbstractNeuralLyapunovStructure,
        θ,
        f;
        p = SciMLBase.NullParameters(),
        jac = _forward_jacobian,
        J_phi = nothing,
        fixed_point = nothing
    )
    u_dim = get_control_dim(structure)
    phi_dim = get_network_dim(structure) - u_dim
    smodel = phi_to_net(phi, θ; idx = 1:phi_dim)
    control_smodel = phi_to_net(phi, θ; idx = (phi_dim + 1):(phi_dim + u_dim))
    if isnothing(J_phi)
        smodel_jac = Base.Fix1(jac, smodel)
    else
        smodel_jac = J_phi
    end

    return StructuredNumericLyapunovControlFunction(
        smodel,
        smodel_jac,
        safe_get_device(θ),
        f,
        p,
        fixed_point,
        control_smodel,
        get_control_structure(structure),
        get_V̇(structure)
    )
end

function (V̇::StructuredNumericLyapunovControlFunction)(x::AbstractVector)
    x0 = isnothing(V̇.fixed_point) ? zero(x) : V̇.fixed_point
    u = V̇.control_structure(V̇.usmodel, x, x0)
    ẋ = V̇.dynamics(x, u, V̇.p, x0)
    dev = V̇.devθ
    return V̇.V̇_structure(V̇.smodel, V̇.Jsmodel, dev(x), dev(ẋ), dev(x0))
end

function (V̇::StructuredNumericLyapunovControlFunction)(x::AbstractMatrix)
    return mapslices(V̇, x, dims = [1])
end


struct StructuredNumericLyapunovDecreaseFunction{S <: StatefulLuxLayer, JS, DΘ, F, P, X0 <: Union{Nothing, AbstractVector{<:Real}}, DV}
    smodel::S
    smodel_jac::JS
    devθ::DΘ
    dynamics::F
    p::P
    fixed_point::X0
    V̇_structure::DV
end

function StructuredNumericLyapunovDecreaseFunction(
        phi::Union{<:Phi, AbstractVector{<:Phi}},
        structure::AbstractNeuralLyapunovStructure,
        θ,
        dynamics;
        p = SciMLBase.NullParameters(),
        jac = _forward_jacobian,
        J_phi = nothing,
        fixed_point = nothing
    )
    smodel = phi_to_net(phi, θ)
    if isnothing(J_phi)
        smodel_jac = Base.Fix1(jac, smodel)
    else
        smodel_jac = J_phi
    end

    V̇_structure = get_V̇(structure)

    return StructuredNumericLyapunovDecreaseFunction(
        smodel,
        smodel_jac,
        safe_get_device(θ),
        dynamics,
        p,
        fixed_point,
        V̇_structure
    )
end

function (V̇::StructuredNumericLyapunovDecreaseFunction)(x::AbstractVector)
    x0 = isnothing(V̇.fixed_point) ? zero(x) : V̇.fixed_point
    ẋ = V̇.dynamics(x, V̇.p, x0)
    dev = V̇.devθ
    return V̇.V̇_structure(V̇.smodel, V̇.smodel_jac, dev(x), dev(ẋ), dev(x0))
end

function (V̇::StructuredNumericLyapunovDecreaseFunction)(x::AbstractMatrix)
    return mapslices(V̇, x, dims = [1])
end

struct ADNumericLyapunovDecreaseFunction{TV, F, P, D}
    V::TV
    dynamics::F
    p::P
    deriv::D
end

function ADNumericLyapunovDecreaseFunction(
        V, f; p = SciMLBase.NullParameters(), deriv = _forward_derivative
    )
    return ADNumericLyapunovDecreaseFunction(V, f, p, deriv)
end


function (V̇::ADNumericLyapunovDecreaseFunction)(x::AbstractVector)
    t0 = zero(eltype(x))
    ẋ = V̇.dynamics(x, V̇.p, t0)
    return V̇.deriv(δt -> V̇.V(x + δt * ẋ), t0)
end

function (V̇::ADNumericLyapunovDecreaseFunction)(x::AbstractMatrix)
    t0 = zero(eltype(x))
    ẋ = mapslices(x, dims = 1) do _x
        return V̇.dynamics(_x, V̇.p, t0)
    end
    return V̇.deriv(δt -> V̇.V(x + δt * ẋ), t0)
end

struct ADNumericLyapunovControlFunction{TV, F, P, X0, D, US, U, DΘ}
    V::TV
    dynamics::F
    p::P
    fixed_point::X0
    deriv::D
    control_smodel::US
    control_structure::U
    devθ::DΘ
end

function ADNumericLyapunovControlFunction(
        V, f, control_smodel, control_structure; p = SciMLBase.NullParameters(),
        fixed_point = nothing, deriv = _forward_derivative
    )
    devθ = safe_get_device(control_smodel)

    return ADNumericLyapunovControlFunction(
        V, f, p, fixed_point, deriv, control_smodel, control_structure, devθ
    )
end

function (V̇::ADNumericLyapunovControlFunction)(x::AbstractVector)
    t0 = zero(eltype(x))
    x0 = isnothing(V̇.fixed_point) ? zero(x) : V̇.fixed_point

    devθ = V̇.devθ
    devx = safe_get_device(x)
    u = devx(V̇.control_structure(V̇.control_smodel, devθ(x), devθ(x0)))
    ẋ = V̇.dynamics(x, u, V̇.p, t0)

    return V̇.deriv(δt -> V̇.V(x + δt * ẋ), t0)
end

function (V̇::ADNumericLyapunovControlFunction)(x::AbstractMatrix)
    t0 = zero(eltype(x))
    x0 = isnothing(V̇.fixed_point) ? zero(x) : V̇.fixed_point

    devθ = V̇.devθ
    devx = safe_get_device(x)

    ẋ = mapslices(x, dims = 1) do _x
        u = devx(V̇.control_structure(V̇.control_smodel, devθ(_x), devθ(x0)))
        return V̇.dynamics(_x, u, V̇.p, t0)
    end

    return V̇.deriv(δt -> V̇.V(x + δt * ẋ), t0)
end
