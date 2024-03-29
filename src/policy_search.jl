"""
    add_policy_search(lyapunov_structure, new_dims, control_structure)

Adds dependence on the neural network to the dynamics in a `NeuralLyapunovStructure`

Adds `new_dims` outputs to the neural network and feeds them through `control_structure` to
calculatethe contribution of the neural network to the dynamics.
The existing `lyapunov_structure.network_dim` dimensions are used as in `lyapunov_structure`
to calculate the Lyapunov function.

`lyapunov_structure` should assume in its `V̇` that the dynamics take a form `f(x, p, t)`.
The returned `NeuralLyapunovStructure` will assume instead `f(x, u, p, t)`, where `u` is the
contribution from the neural network. Therefore, this structure cannot be used with a
`NeuralLyapunovPDESystem` method that requires an `ODEFunction`, `ODESystem`, or
`ODEProblem`.
"""
function add_policy_search(
        lyapunov_structure::NeuralLyapunovStructure,
        new_dims::Integer;
        control_structure::Function = identity
)::NeuralLyapunovStructure
    let V = lyapunov_structure.V, ∇V = lyapunov_structure.∇V, V̇ = lyapunov_structure.V̇,
        V_dim = lyapunov_structure.network_dim, nd = new_dims, u = control_structure

        NeuralLyapunovStructure(
            function (net, state, fixed_point)
                if length(size(state)) == 1
                    if V_dim == 1
                        V(st -> net(st)[1], state, fixed_point)
                    else
                        V(st -> net(st)[1:V_dim], state, fixed_point)
                    end
                else
                    V(st -> net(st)[1:V_dim, :], state, fixed_point)
                end
            end,
            function (net, J_net, state, fixed_point)
                ∇V(st -> net(st)[1:V_dim], st -> J_net(st)[1:V_dim, :], state, fixed_point)
            end,
            function (net, J_net, f, state, params, t, fixed_point)
                V̇(st -> net(st)[1:V_dim], st -> J_net(st)[1:V_dim, :],
                    (st, p, t) -> f(st, u(net(st)[(V_dim + 1):end]), p, t), state, params,
                    t, fixed_point)
            end,
            (f, net, state, p, t) -> f(state, u(net(state)[(V_dim + 1):end]), p, t),
            V_dim + nd
        )
    end
end

"""
    get_policy(phi, θ, network_func, dim; control_structure)

Returns the control policy as a function of the state

The returned function can operate on a state vector or columnwise on a matrix of state
vectors.

`phi` is the neural network with parameters `θ`. `network_func` is an output of
`NeuralLyapunovPDESystem`.
The control policy is `control_structure` composed with the last `dim` outputs of the
neural network, as set up by `add_policy_search`.
"""
function get_policy(
        phi,
        θ,
        network_func::Function,
        dim::Integer;
        control_structure::Function = identity
)
    function policy(state::AbstractVector)
        control_structure(network_func(phi, θ, state)[(end - dim + 1):end])
    end

    policy(state::AbstractMatrix) = mapslices(policy, state, dims = [1])

    return policy
end
