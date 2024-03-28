"""
    get_numerical_lyapunov_function(phi, θ, structure, dynamics, fixed_point;
                                    <keyword_arguments>)

Combine Lyapunov function structure, dynamics, and neural network weights to generate Julia
functions representing the Lyapunov function and its time derivative: ``V(x), V̇(x)``.

These functions can operate on a state vector or columnwise on a matrix of state vectors.

# Arguments
- `phi`, `θ`: `phi` is the neural network with parameters `θ`.
- `structure`: a [`NeuralLyapunovStructure`](@ref) representing the structure of the neural
        Lyapunov function.
- `dynamics`: the system dynamics, as a function to be used in conjunction with
        `structure.f_call`.
- `fixed_point`: the equilibrium point being analyzed by the Lyapunov function.
- `p`: parameters to be passed into `dynamics`; defaults to `SciMLBase.NullParameters()`.
- `use_V̇_structure`: when `true`, ``V̇(x)`` is calculated using `structure.V̇`; when `false`,
        ``V̇(x)`` is calculated using `deriv` as ``\\frac{d}{dt} V(x + t f(x))`` at
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
        dynamics::Function,
        fixed_point;
        p = SciMLBase.NullParameters(),
        use_V̇_structure = false,
        deriv = ForwardDiff.derivative,
        jac = ForwardDiff.jacobian,
        J_net = nothing
)::Tuple{Function, Function}
    # network_func is the numerical form of neural network output
    output_dim = structure.network_dim
    network_func = let φ = phi, _θ = θ, dim = output_dim
        function (x)
            reduce(
                vcat,
                Array(φ[i](x, _θ.depvar[Symbol(:φ, i)])) for i in 1:dim
            )
        end
    end

    # V is the numerical form of Lyapunov function
    V = let V_structure = structure.V, net = network_func, x0 = fixed_point
        V(state::AbstractVector) = V_structure(net, state, x0)
        V(state::AbstractMatrix) = mapslices(V, state, dims = [1])
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
            V̇(state::AbstractMatrix) = mapslices(V̇, state, dims = [1])

            return _V, V̇
        end
    else
        let f_call = structure.f_call, _V = V, net = network_func, f = dynamics, params = p,
            _deriv = deriv

            # Numerical time derivative of Lyapunov function
            V̇(state::AbstractVector) = _deriv(
                (δt) -> _V(state + δt * f_call(f, net, state, params, 0.0)),
                0.0
            )
            V̇(state::AbstractMatrix) = mapslices(V̇, state, dims = [1])

            return _V, V̇
        end
    end
end
