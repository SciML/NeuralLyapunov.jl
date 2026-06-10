
"""
    NeuralLyapunovControlStructure(V, V̇, control_structure, network_dim, control_dim)

Specifies the structure of the neural Lyapunov function and its derivative.

Allows the user to define the Lyapunov in terms of the neural network, potentially
structurally enforcing some Lyapunov conditions.

# Fields
  - `V(phi, state, fixed_point)`: outputs the value of the Lyapunov function at `state`.
  - `V̇(phi, J_phi, state, dstate_dt, fixed_point)`: outputs the time derivative of
    the Lyapunov function at `state`.
  - `control_structure(phi_c, state, fixed_point)`: transforms the final `control_dim`
    outputs of the neural net before passing them as `u` into the dynamics `f(x, u, p, t)`.
  - `network_dim`: the dimension of the output of the neural network.
  - `control_dim`: the number of neural network outputs used in the control policy.

`phi` and `J_phi` above are both functions of `state` alone.
"""
struct NeuralLyapunovControlStructure{TV, TDV, U, D <: Integer, C <: Integer} <: AbstractNeuralLyapunovStructure{true}
    V::TV
    V̇::TDV
    control_structure::U
    network_dim::D
    control_dim::C
end

get_V(spec::NeuralLyapunovControlStructure) = spec.V
get_V̇(spec::NeuralLyapunovControlStructure) = spec.V̇
get_network_dim(spec::NeuralLyapunovControlStructure) = spec.network_dim
get_control_structure(spec::NeuralLyapunovControlStructure) = spec.control_structure
get_control_dim(spec::NeuralLyapunovControlStructure) = spec.control_dim

function Base.show(io::IO, s::NeuralLyapunovControlStructure)
    n = s.network_dim
    @variables φ_V(..) ∇φ_V(..) φ_c(..) ∇φ_c(..) x ẋ x_0 p t
    println(io, "NeuralLyapunovControlStructure")
    println(io, "    Network dimension: ", n)
    try
        V = string(s.V(φ_V, x, x_0))
        # Replace abs2(A) with ||A||² for better readability
        V = replace(V, r"abs2\(\s*((?:[^()]+|\((?1)\))*)\s*\)" => s"||\1||²")
        # Replace ^2 with ²
        V = replace(V, r"\^2" => "²")
        println(io, "    V(x) = ", V)
    catch e
        println(io, "    V(x) = <could not display: $(e)>")
    end
    try
        V̇ = string(s.V̇(φ_V, ∇φ_V, x, ẋ, x_0))
        # Replace abs2(A) with ||A||^2 for better readability
        V̇ = replace(V̇, r"abs2\(\s*((?:[^()]+|\((?1)\))*)\s*\)" => s"||\1||²")
        # Replace ^2 with ²
        V̇ = replace(V̇, r"\^2" => "²")
        # Replace Differential(x, 1)(φ(x)) with ∇φ(x) for better readability
        V̇ = replace(V̇, r"Differential\(x, 1\)\(φ\(x\)\)" => "∇φ(x)")
        println(io, "    V̇(x) = ", V̇)
    catch e
        println(io, "    V̇(x) = <could not display: $(e)>")
    end
    try
        u = string(s.control_structure(φ_c, x, x_0))
        # Replace abs2(A) with ||A||^2 for better readability
        u = replace(u, r"abs2\(\s*((?:[^()]+|\((?1)\))*)\s*\)" => s"||\1||²")
        # Replace ^2 with ²
        u = replace(u, r"\^2" => "²")
        println(io, "    u(x) = ", u)
    catch e
        println(io, "    u(x) = <could not display: $(e)>")
    end
    return
end

"""
    add_policy_search(lyapunov_structure, new_dims; control_structure)

Add dependence on the neural network to the dynamics in a [`NeuralLyapunovStructure`](@ref).

# Arguments
  - `lyapunov_structure::NeuralLyapunovStructure`: provides structure for ``V, V̇``; should
    assume dynamics take a form of `f(x, p, t)`.
  - `new_dims::Integer`: number of outputs of the neural network to pass into the dynamics
    through `control_structure`.

# Keyword Arguments
  - `control_structure`: function that transforms the final `new_dims` outputs of the neural
    network before passing them into the dynamics; defaults to `identity`, passing in the
    neural network outputs unchanged.

The returned `NeuralLyapunovStructure` expects dynamics of the form `f(x, u, p, t)`, where
`u` captures the dependence of dynamics on the neural network (e.g., through a control
input). When evaluating the dynamics, it uses `u = control_structure(phi_end(x))` where
`phi_end` is a function that returns the final `new_dims` outputs of the neural network.
The other `lyapunov_structure.network_dim` outputs are used for calculating ``V`` and ``V̇``,
as specified originally by `lyapunov_structure`.

```jldoctest; filter = [r"(ẋ|ẋ)" => "y", r"\\(x\\)" => "", r"\\s*\\*\\s*" => "", r"∇φ_V" => "∇", r"(?m)\\s*V̇\\(x\\)\\s*=\\s*(2|∇|φ_V|y){4}\$"]
add_policy_search(NonnegativeStructure(3), 1)
# output
NeuralLyapunovControlStructure
    Network dimension: 4
    V(x) = φ_V(x)²
    V̇(x) = 2∇φ_V(x)*φ_V(x)*ẋ
    u(x) = φ_c(x)
```
"""
function add_policy_search(
        lyapunov_structure::AbstractNeuralLyapunovStructure{false},
        new_dims::Integer;
        control_structure = (phi, x, x0) -> phi(x)
    )::NeuralLyapunovControlStructure
    let V = get_V(lyapunov_structure), V̇ = get_V̇(lyapunov_structure),
            V_dim = get_network_dim(lyapunov_structure), u = control_structure
        return NeuralLyapunovControlStructure(V, V̇, u, V_dim + new_dims, new_dims)
    end
end

"""
    get_policy(phi, θ, network_dim, control_dim; fixed_point, control_structure)
    get_policy(phi, θ, structure::AbstractNeuralLyapunovStructure{true}; fixed_point)

Generate a Julia function representing the control policy/unmodeled portion of the dynamics
as a function of the state.

The returned function can operate on a state vector or columnwise on a matrix of state
vectors.

# Positional Arguments
  - `phi`: the neural network, represented as `phi(state, θ)` if the neural network has a
    single output, or a `Vector` of the same with one entry per neural network output.
  - `θ`: the parameters of the neural network; `θ[:φ1]` should be the parameters of the
    first neural network output (even if there is only one), `θ[:φ2]` the parameters of the
    second (if there are multiple), and so on.
  - `network_dim`: total number of neural network outputs.
  - `control_dim`: number of neural network outputs used in the control policy.
  - `structure::AbstractNeuralLyapunovStructure{true}`: provides the control structure and
    dimensions for the neural network outputs used in the control policy.

# Keyword Arguments
  - `fixed_point`: the fixed point of the system.
  - `control_structure`: transforms the final `control_dim` outputs of the neural net before
    passing them into the dynamics; defaults to `identity`, passing in the neural network
    outputs unchanged.
"""
function get_policy(
        phi,
        θ,
        network_dim::Integer,
        control_dim::Integer;
        fixed_point = nothing,
        control_structure = (phi, x, x0) -> phi(x)
    )
    network_func = phi_to_net(phi, θ; idx = (network_dim - control_dim + 1):network_dim)

    function policy(state::AbstractVector)
        return control_structure(
            network_func,
            state,
            isnothing(fixed_point) ? zero(state) : fixed_point
        )
    end
    function policy(states::AbstractMatrix)
        return mapslices(policy, states, dims = [1])
    end

    return policy
end

function get_policy(
    phi,
    θ,
    structure::AbstractNeuralLyapunovStructure{true};
    fixed_point = nothing
)
    return get_policy(
        phi,
        θ,
        get_network_dim(structure),
        get_control_dim(structure);
        control_structure = get_control_structure(structure),
        fixed_point
    )
end
