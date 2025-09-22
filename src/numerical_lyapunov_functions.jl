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
  - `dynamics`: the system dynamics, as a function to be used in conjunction with
    `structure.f_call`.
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
        structure::NeuralLyapunovStructure,
        dynamics,
        fixed_point;
        p = SciMLBase.NullParameters(),
        use_V̇_structure = false,
        deriv = ForwardDiff.derivative,
        jac = ForwardDiff.jacobian,
        J_net = nothing
)
    # network_func is the numerical form of neural network output
    network_func = phi_to_net(phi, θ)

    # V is the numerical form of Lyapunov function
    V = let V_structure = structure.V, net = network_func, x0 = fixed_point
        V(state::AbstractVector) = V_structure(net, state, x0)
        V(states::AbstractMatrix) = mapslices(V, states, dims = [1])
    end

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

        let _V = V, V̇_structure = structure.V̇, net = network_func, x0 = fixed_point,
            f = dynamics, params = p, _J_net = network_jacobian

            # Numerical time derivative of Lyapunov function
            V̇(state::AbstractVector) = V̇_structure(net, _J_net, f, state, params, 0.0, x0)
            V̇(states::AbstractMatrix) = mapslices(V̇, states, dims = [1])

            return _V, V̇
        end
    else
        let f_call = structure.f_call, _V = V, net = network_func, f = dynamics, params = p,
            _deriv = deriv

            # Numerical time derivative of Lyapunov function
            function V̇(state::AbstractVector)
                return _deriv(
                    δt -> _V(state + δt * f_call(f, net, state, params, 0.0)),
                    0.0
                )
            end
            function V̇(states::AbstractMatrix)
                ẋ = mapslices(states, dims = [1]) do state
                    return f_call(f, net, state, params, 0.0)
                end
                return _deriv(δt -> _V(states + δt * ẋ), 0.0)
            end

            return _V, V̇
        end
    end
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
            reduce(
                vcat,
                Array(φ[i](x, _θ[Symbol(:φ, i)])) for i in _idx
            )
        end
    end
end
