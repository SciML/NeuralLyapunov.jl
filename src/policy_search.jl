"""
    add_policy_search(lyapunov_structure, new_dims, control_structure)

Add dependence on the neural network to the dynamics in a `NeuralLyapunovStructure`.

Add `new_dims` outputs to the neural network and feeds them through `control_structure` to
calculate the contribution of the neural network to the dynamics.
Use the existing `lyapunov_structure.network_dim` dimensions as in `lyapunov_structure` to
calculate the Lyapunov function.

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

Generate a Julia function representing the control policy as a function of the state

The returned function can operate on a state vector or columnwise on a matrix of state
vectors.

`phi` is the neural network with parameters `θ`. The network should have `network_dim`
outputs, the last `control_dim` of which will be passed into `control_structure` to create
the policy output.
"""
function get_policy(
        phi,
        θ,
        network_dim::Integer,
        control_dim::Integer;
        control_structure::Function = identity
)
    function policy(state::AbstractVector)
        control_structure(
            reduce(
                vcat,
                Array(phi[i](state, θ.depvar[Symbol(:φ, i)]))
                    for i in (network_dim - control_dim + 1):network_dim
            )
        )
    end

    policy(state::AbstractMatrix) = mapslices(policy, state, dims = [1])

    return policy
end
