module NeuralLyapunov

import ForwardDiff
using ModelingToolkit
using LinearAlgebra

export NeuralLyapunovPDESystem, NumericalNeuralLyapunovFunctions

function NeuralLyapunovPDESystem(dynamics::Function, lb, ub, output_dim::Integer=1; δ::Real=0.01, relu=(t)->max(0.0,t), fixed_point=nothing)::Tuple{PDESystem, Function}
    # Define state symbols
    state_dim = lb isa AbstractArray ? length(lb) : ub isa AbstractArray ? length(ub) : 1
    state_syms = [Symbol(:state, i) for i in 1:state_dim]
    state = [first(@parameters $s) for s in state_syms]

    # Define domains
    if !(lb isa AbstractArray)
        lb = ones(state_dim) .* lb
    end
    if !(ub isa AbstractArray)
        ub = ones(state_dim) .* ub
    end
    domains = [ state[i] ∈ (lb[i], ub[i]) for i in 1:state_dim ]

    # Define Lyapunov function
    net_syms = [Symbol(:u, i) for i in 1:output_dim]
    net = [first(@variables $s(..)) for s in net_syms]
    "Symbolic form of neural network output"
    u(x) = Num.([ui(x...) for ui in net])
    fixed_point = isnothing(fixed_point) ? zeros(state_dim) : fixed_point
    "Symobolic form of the Lyapunov function"
    V_sym(x) = ( u(x) - u(fixed_point) ) ⋅ ( u(x) - u(fixed_point) ) + δ*log( 1. + ( x - fixed_point ) ⋅ ( x - fixed_point ) )

    # Define dynamics and Lyapunov conditions
    "Symbolic time derivative of the Lyapunov function"
    V̇_sym(x) = dynamics(x) ⋅ Symbolics.gradient(V_sym(x), x)
    "V̇ should be negative"
    eq = relu(V̇_sym(state)) ~ 0.0

    # Construct PDESystem
    "V should be 0 at the fixed point"
    bcs = [ V_sym(fixed_point) ~ 0.0 ]
    @named lyapunov_pde_system = PDESystem(eq, bcs, domains, state, u(state))

    # Make Lyapunov function 
    "Numerical form of neural network output"
    u_func(phi, res, x) = reduce( vcat, phi[i](x, res.u.depvar[net_syms[i]]) for i in 1:output_dim )

    "Numerical form of Lyapunov function"
    function V_func(phi, res, x) 
        u_vec = u_func(phi, res, x) .- u_func(phi, res, fixed_point)
        u2 = mapslices(norm, u_vec, dims=[1]).^2
        l = δ*log.(1.0 .+ mapslices(norm, x, dims=[1]).^2)
        u2 .+ l
    end

    return lyapunov_pde_system, V_func
end

function NeuralLyapunovPDESystem(dynamics::ODEProblem, lb, ub, output_dim::Integer=1; δ::Real=0.01, relu=(t)->max(0.0,t), fixed_point=nothing)::Tuple{PDESystem, Function}
    f = get_dynamics_from_ODEProblem(dynamics)
    return NeuralLyapunovPDESystem(f, lb, ub, output_dim; δ, relu, fixed_point)
end

function NumericalNeuralLyapunovFunctions(phi, result, lyapunov_func, dynamics::Function; grad=ForwardDiff.gradient)
    "Numerical form of Lyapunov function"
    V_func(state::AbstractMatrix) = lyapunov_func(phi, result, state)
    V_func(state::AbstractVector) = first(lyapunov_func(phi, result, state))

    "Numerical gradient of Lyapunov function"
    ∇V_func(state::AbstractVector) = grad(V_func, state)
    ∇V_func(state::AbstractMatrix) = mapslices(∇V_func, state, dims=[1])

    "Numerical time derivative of Lyapunov function"
    V̇_func(state::AbstractVector) = dynamics(state) ⋅ ∇V_func(state)
    V̇_func(state::AbstractMatrix) = reshape(map(x->x[1]⋅x[2], zip(eachslice(dynamics(state), dims=2), eachslice(∇V_func(state), dims=2))), (1,:))

    return V_func, V̇_func
end

function NumericalNeuralLyapunovFunctions(phi, result, lyapunov_func, dynamics::ODEProblem; grad=ForwardDiff.gradient)
    f = get_dynamics_from_ODEProblem(dynamics)
    return NumericalNeuralLyapunovFunctions(phi, result, lyapunov_func, f; grad)
end

function get_dynamics_from_ODEProblem(prob::ODEProblem)::Function
    dynamicsODEfunc = prob.f
    f_ = if dynamicsODEfunc.mass_matrix == I
        state -> dynamicsODEfunc.f(state, prob.p, 0.0) # Let time be 0.0, since we're only considering time-invariant dynamics
    else
        throw(ErrorException("For now, we only accept ODEs, not DAEs"))
    end
    f(state::AbstractVector) = f_(state)
    f(state::AbstractMatrix) = mapslices(f_, state, dims=[1])
    return f
end

end