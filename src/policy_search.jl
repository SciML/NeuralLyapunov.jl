"""
    add_policy_search(lyapunov_structure, new_dims; control_structure)

Add dependence on the neural network to the dynamics in a [`NeuralLyapunovStructure`](@ref).

# Arguments
  - `lyapunov_structure::NeuralLyapunovStructure`: provides structure for ``V, V̇``; should
    assume dynamics take a form of `f(x, p, t)`.
  - `new_dims::Integer`: number of outputs of the neural network to pass into the dynamics
    through `control_structure`.

# Keyword Arguments
  - `control_structure::Function`: transforms the final `new_dims` outputs of the neural net
    before passing them into the dynamics; defaults to `identity`, passing in the neural
    network outputs unchanged.

The returned `NeuralLyapunovStructure` expects dynamics of the form `f(x, u, p, t)`, where
`u` captures the dependence of dynamics on the neural network (e.g., through a control
input). When evaluating the dynamics, it uses `u = control_structure(phi_end(x))` where
`phi_end` is a function that returns the final `new_dims` outputs of the neural network.
The other `lyapunov_structure.network_dim` outputs are used for calculating ``V`` and ``V̇``,
as specified originally by `lyapunov_structure`.
"""
function add_policy_search(
        lyapunov_structure::NeuralLyapunovStructure,
        new_dims::Integer;
        control_structure::Function = identity
)::NeuralLyapunovStructure
    let V = lyapunov_structure.V, V̇ = lyapunov_structure.V̇,
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
    get_policy(phi, θ, network_dim, control_dim; control_structure)

Generate a Julia function representing the control policy/unmodeled portion of the dynamics
as a function of the state.

The returned function can operate on a state vector or columnwise on a matrix of state
vectors.

# Arguments
  - `phi`: the neural network, represented as `phi(state, θ)` if the neural network has a
    single output, or a `Vector` of the same with one entry per neural network output.
  - `θ`: the parameters of the neural network; `θ[:φ1]` should be the parameters of the
    first neural network output (even if there is only one), `θ[:φ2]` the parameters of the
    second (if there are multiple), and so on.
  - `network_dim`: total number of neural network outputs.
  - `control_dim`: number of neural network outputs used in the control policy.

# Keyword Arguments
  - `control_structure`: transforms the final `control_dim` outputs of the neural net before
    passing them into the dynamics; defaults to `identity`, passing in the neural network
    outputs unchanged.
"""
function get_policy(
        phi,
        θ,
        network_dim::Integer,
        control_dim::Integer;
        control_structure::Function = identity
)
    network_func = phi_to_net(phi, θ; idx = (network_dim - control_dim + 1):network_dim)

    function policy(state::AbstractVector)
        control_structure(network_func(state))
    end
    function policy(states::AbstractMatrix)
        mapslices(
            control_structure,
            network_func(states),
            dims = [1]
        )
    end

    return policy
end
