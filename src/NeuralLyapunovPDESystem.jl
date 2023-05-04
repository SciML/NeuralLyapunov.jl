function NeuralLyapunovPDESystem(
    dynamics::ODEFunction,
    lb,
    ub;
    default_ps = SciMLBase.NullParameters(),
    output_dim::Integer = 1,
    δ::Real = 0.01,
    ϵ::Real = 0.01,
    relu = (t) -> max(0.0, t),
    fixed_point = nothing,
)::Tuple{PDESystem,Function}
    if dynamics.mass_matrix !== I
        throw(ErrorException("DAEs are not supported at this time"))
    end
    state_dim = length(lb)

    # Define state symbols, if not already defined
    state_syms = if isnothing(dynamics.syms)
        [Symbol(:state, i) for i = 1:state_dim]  
    else
        dynamics.syms
    end
    state = [first(@parameters $s) for s in state_syms]

    # Define parameter symbols, if not already defined
    param_syms, default_ps = if default_ps == SciMLBase.NullParameters()
        [], default_ps
    elseif isa(default_ps, Dict)
        [first(_pair) for _pair in default_ps], default_ps
    elseif isa(default_ps, Vector) && isa(first(default_ps), Pair)
        first.(default_ps), Dict(default_ps)
    elseif isa(default_ps, Vector)
        if isnothing(dynamics.paramsyms)
            syms = [Symbol(:param, i) for i = 1:length(default_ps)]
            syms, Dict([ sym => val for (sym, val) in zip(syms, default_ps)])
        else
            dynamics.paramsyms, Dict([ sym => val for (sym, val) in zip(dynamics.paramsyms, default_ps)])
        end
    else
        throw(ErrorException("Default parameters have unsupported type. default_ps should be NullParameters, Dict, Vector{Pair}, or Vector of values"))
    end
    
    params = [first(@parameters $s) for s in param_syms]
    defaults = if default_ps == SciMLBase.NullParameters()
        default_ps
    else
        Dict([ param => default_ps[param_sym] for (param, param_sym) in zip(params, param_syms) ])
    end

    # Define domains
    domains = [state[i] ∈ (lb[i], ub[i]) for i = 1:state_dim]

    # Define Lyapunov function
    net_syms = [Symbol(:u, i) for i = 1:output_dim]
    net = [first(@variables $s(..)) for s in net_syms]
    # u(x) is the symbolic form of neural network output
    u(x) = [Num(ui(x...)) for ui in net]
    fixed_point = isnothing(fixed_point) ? zeros(state_dim) : fixed_point
    # V_sym(x) is the symobolic form of the Lyapunov function
    V_sym(x) =
        (u(x) - u(fixed_point)) ⋅ (u(x) - u(fixed_point)) +
        δ * log(1.0 + (x - fixed_point) ⋅ (x - fixed_point))

    # Define dynamics and Lyapunov conditions
    # V̇_sym(x) is the symbolic time derivative of the Lyapunov function
    V̇_sym(x) = dynamics(x, params, 0.0) ⋅ Symbolics.gradient(V_sym(x), x)
    # V̇ should be negative when V < 1, and try not to let V >> 1
    eqs = [
        relu(V̇_sym(state) + ϵ * (state - fixed_point) ⋅ (state - fixed_point)) * relu(1 - V_sym(state)) ~ 0.0,
        relu(V_sym(state) - 1) ~ 0.0,
    ]
    #eqs = [ relu(V̇_sym(state)+ϵ*(state - fixed_point)⋅(state - fixed_point)) ~ 0.0 ]

    # Construct PDESystem
    bcs = vcat(
        collect(
            V_sym(vcat(state[1:i-1], lb[i], state[i+1:end])) ~ 1.1 for i = 1:state_dim
        ),
        collect(
            V_sym(vcat(state[1:i-1], ub[i], state[i+1:end])) ~ 1.1 for i = 1:state_dim
        ),
    )
    bcs = [V_sym(fixed_point) ~ 0.0] # V should be 0 at the fixed point
    @named lyapunov_pde_system = PDESystem(eqs, bcs, domains, state, u(state), params, defaults=defaults)

    # Make Lyapunov function 
    # u_func is the numerical form of neural network output
    u_func(phi, res, x) =
        reduce(vcat, Array(phi[i](x, res.u.depvar[net_syms[i]])) for i = 1:output_dim)

    """
        V_func(phi, res, x)
    Numerical form of the Lyapunov function.

    Evaluates the Lyapunov function using the neural net phi and parameters res
    at the state x. If x is a matrix of states, V_func operates columnwise.
    """
    function V_func(phi, res, x)
        u_vec = u_func(phi, res, x) .- u_func(phi, res, fixed_point)
        u2 = mapslices(norm, u_vec, dims = [1]) .^ 2
        l = δ * log.(1.0 .+ mapslices(norm, x .- fixed_point, dims = [1]) .^ 2)
        u2 .+ l
    end

    return lyapunov_pde_system, V_func
end

function NeuralLyapunovPDESystem(
    dynamics::Function,
    lb,
    ub,
    output_dim::Integer = 1;
    δ::Real = 0.01,
    ϵ::Real = 0.01,
    relu = (t) -> max(0.0, t),
    fixed_point = nothing,
)::Tuple{PDESystem,Function}
    return NeuralLyapunovPDESystem(
            ODEFunction(dynamics),
            lb,
            ub,
            output_dim;
            δ,
            ϵ,
            relu,
            fixed_point,
        )      
end

function NeuralLyapunovPDESystem(
    dynamics::ODEProblem,
    lb,
    ub,
    output_dim::Integer = 1;
    δ::Real = 0.01,
    relu = (t) -> max(0.0, t),
    fixed_point = nothing,
)::Tuple{PDESystem,Function}
    f = get_dynamics_from_ODEProblem(dynamics)
    return NeuralLyapunovPDESystem(f, lb, ub, output_dim; δ, relu, fixed_point)
end

"""
    NumericalNeuralLyapunovFunctions(phi, result, lyapunov_func, dynamics, grad)
Returns the Lyapunov function, its time derivative, and its gradient: V(state), 
V̇(state)

These functions can operate on a state vector or columnwise on a matrix of state
vectors. Gradients are calculated using grad, which defaults to ForwardDiff.gradient.
phi is the neural network with parameters given by result. lyapunov_func is an 
output of NeuralLyapunovPDESystem.
"""
function NumericalNeuralLyapunovFunctions(
    phi,
    result,
    lyapunov_func,
    dynamics::Function;
    grad = ForwardDiff.gradient,
)::Tuple{Function, Function, Function}
    # Numerical form of Lyapunov function
    V_func(state::AbstractMatrix) = lyapunov_func(phi, result, state)
    V_func(state::AbstractVector) = first(lyapunov_func(phi, result, state))

    # Numerical gradient of Lyapunov function
    ∇V_func(state::AbstractVector) = grad(V_func, state)
    ∇V_func(state::AbstractMatrix) = mapslices(∇V_func, state, dims = [1])

    # Numerical time derivative of Lyapunov function
    V̇_func(state::AbstractVector) = dynamics(state) ⋅ ∇V_func(state)
    V̇_func(state::AbstractMatrix) = reshape(
        map(
            x -> x[1] ⋅ x[2],
            zip(eachslice(dynamics(state), dims = 2), eachslice(∇V_func(state), dims = 2)),
        ),
        (1, :),
    )

    return V_func, V̇_func, ∇V_func
end

function NumericalNeuralLyapunovFunctions(
    phi,
    result,
    lyapunov_func,
    dynamics::ODEProblem;
    grad = ForwardDiff.gradient,
)::Tuple{Function, Function, Function}
    f = get_dynamics_from_ODEProblem(dynamics)
    return NumericalNeuralLyapunovFunctions(phi, result, lyapunov_func, f; grad)
end


"""
    get_dynamics_from_ODEProblem(prob)
Extracts f such that ODEProblem is ẋ = f(x)

The returned function f can operate on a single x vector or columnwise on a 
matrix of x values.
"""
function get_dynamics_from_ODEProblem(prob::ODEProblem)::Function
    dynamicsODEfunc = prob.f
    f_ = if dynamicsODEfunc.mass_matrix == I
        state -> dynamicsODEfunc.f(state, prob.p, 0.0) # Let time be 0.0, since we're only considering time-invariant dynamics
    else
        throw(ErrorException("DAEs are not supported at this time"))
    end
    f(state::AbstractVector) = f_(state)
    f(state::AbstractMatrix) = mapslices(f_, state, dims = [1])
    return f
end
