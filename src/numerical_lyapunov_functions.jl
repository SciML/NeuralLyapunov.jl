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
  - `phi`: the neural network, represented as `phi(x, θ)` if the neural network has a single
    output, or a `Vector` of the same with one entry per neural network output.
  - `θ`: the parameters of the neural network; If the neural network has multiple outputs,
    `θ[:φ1]` should be the parameters of the first neural network output, `θ[:φ2]` the
    parameters of the second (if there are multiple), and so on. If the neural network has a
    single output, `θ` should be the parameters of the network.
  - `structure`: a [`NeuralLyapunovStructure`](@ref) representing the structure of the
    neural Lyapunov function.
  - `dynamics`: the system dynamics, as a function `ẋ = f(x[, u], p, t)`.
  - `fixed_point`: the equilibrium point being analyzed by the Lyapunov function.

# Keyword Arguments
  - `p`: parameters to be passed into `dynamics`; defaults to `SciMLBase.NullParameters()`.
  - `use_V̇_structure`: when `true`, ``V̇(x)`` is calculated using `structure.V̇`; when `
    false`, ``V̇(x)`` is calculated using `deriv` as ``\\frac{∂}{∂t} V(x + t f(x))`` at
    ``t = 0``; defaults to `false`, as it is more efficient in many cases.
  - `deriv`: a function for calculating derivatives; defaults to (and expects same arguments
    as) `ForwardDiff.derivative`; only used when `use_V̇_structure` is `false`.
  - `jac`: a function for calculating Jacobians; defaults to (and expects same arguments as)
    `ForwardDiff.jacobian`; only used when `use_V̇_structure` is `true`.
  - `J_net`: the Jacobian of the neural network, specified as a function
    `J_net(phi, θ, state)`; if `isnothing(J_net)` (as is the default), `J_net` will be
    calculated using `jac`; only used when `use_V̇_structure` is `true`.
"""
function get_numerical_lyapunov_function(
        phi,
        θ,
        structure::AbstractNeuralLyapunovStructure{nc},
        dynamics,
        fixed_point::AbstractVector;
        p = SciMLBase.NullParameters(),
        use_V̇_structure::Bool = false,
        deriv = ForwardDiff.derivative,
        jac = ForwardDiff.jacobian,
        J_net = nothing
    ) where nc
    # network_func is the numerical form of neural network output
    if nc
        u_dim = get_control_dim(structure)
        φ_dim = get_network_dim(structure) - u_dim
    else
        φ_dim = get_network_dim(structure)
    end
    network_func = phi_to_net(phi, θ; idx = 1:φ_dim)

    # V is the numerical form of Lyapunov function
    V = get_numerical_V(structure.V, network_func, copy(fixed_point))

    if use_V̇_structure
        # Make Jacobian of network_func
        network_jacobian = if isnothing(J_net)
            let net = network_func, J = jac
                (x) -> J(net, x)
            end
        else
            let _J_net = J_net, φ = phi, _θ = θ
                (x) -> _J_net(φ, _θ, x)
            end
        end

        V̇ = if nc
            get_V̇_from_structure(
                structure.V̇,
                network_func,
                network_jacobian,
                dynamics,
                copy(p),
                copy(fixed_point),
                get_control_structure(structure)
            )
        else
            get_V̇_from_structure(
                structure.V̇,
                network_func,
                network_jacobian,
                dynamics,
                copy(p),
                copy(fixed_point)
            )
        end
        return V, V̇
    else
        V̇ = if nc
            control_network = phi_to_net(phi, θ; idx = (φ_dim + 1):(φ_dim + u_dim))
            get_V̇_from_deriv(
                V,
                dynamics,
                copy(p),
                deriv,
                get_control_structure(structure),
                control_network,
                copy(fixed_point)
            )
        else
            get_V̇_from_deriv(V, dynamics, copy(p), deriv)
        end
        return V, V̇
    end
end

function get_numerical_V(V_structure, net, x0)
    V(x::AbstractVector) = V_structure(net, x, x0)
    V(x::AbstractMatrix) = mapslices(V, x, dims = [1])
    return V
end

function get_V̇_from_structure(V̇_structure, net, J_net, f, params, x0)
    # Numerical time derivative of Lyapunov function
    function V̇(x::AbstractVector{T}) where T <: Real
        dstate_dt = f(x, params, zero(T))
        return V̇_structure(net, J_net, x, dstate_dt, x0)
    end
    function V̇(x::AbstractMatrix)
        return mapslices(V̇, x, dims = [1])
    end
    return V̇
end

function get_V̇_from_structure(V̇_structure, net, J_net, f, params, x0, u)
    # Numerical time derivative of Lyapunov function
    function V̇(x::AbstractVector{T}) where T <: Real
        dstate_dt = f(x, u(net, x, x0), params, zero(T))
        return V̇_structure(net, J_net, x, dstate_dt, x0)
    end
    function V̇(x::AbstractMatrix)
        return mapslices(V̇, x, dims = [1])
    end
    return V̇
end

function get_V̇_from_deriv(V, f, p, deriv)
    function V̇(x::AbstractVector{T}) where T <: Real
        return deriv(δt -> V(x + δt * f(x, p, zero(T))), zero(T))
    end
    function V̇(x::AbstractMatrix{T}) where T <: Real
        ẋ = mapslices(x, dims = [1]) do state
            return f(state, p, zero(T))
        end
        return deriv(δt -> V(x + δt * ẋ), zero(T))
    end
    return V̇
end

function get_V̇_from_deriv(V, f, p, deriv, u, u_net, x0)
    function V̇(x::AbstractVector{T}) where T <: Real
        ẋ = f(x, u(u_net, x, x0), p, zero(T))
        return deriv(δt -> V(x + δt * ẋ), zero(T))
    end
    function V̇(x::AbstractMatrix{T}) where T <: Real
        ẋ = mapslices(state -> f(state, u(u_net, state, x0), p, zero(T)), x, dims = [1])
        return deriv(δt -> V(x + δt * ẋ), zero(T))
    end
    return V̇
end

"""
    phi_to_net(phi, θ[; idx])

Return the network as a function of state alone.

# Arguments
  - `phi`: the neural network, represented as `phi(x, θ)` if the neural network has a single
    output, or a `Vector` of the same with one entry per neural network output.
  - `θ`: the parameters of the neural network; If the neural network has multiple outputs,
    `θ[:φ1]` should be the parameters of the first neural network output, `θ[:φ2]` the
    parameters of the second (if there are multiple), and so on. If the neural network has a
    single output, `θ` should be the parameters of the network.
  - `idx`: the neural network outputs to include in the returned function; defaults to all
    and only applicable when `phi isa Vector`.
"""
function phi_to_net(phi, θ)
    return Base.Fix2(phi, θ)
end

function phi_to_net(phi::Vector, θ; idx = eachindex(phi))
    let _θ = θ, φ = phi, _idx = idx
        return function (x)
            return reduce(
                vcat,
                Array(φ[i](x, _θ[Symbol(:φ, i)])) for i in _idx
            )
        end
    end
end
