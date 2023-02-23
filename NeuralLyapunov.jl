module NeuralLyapunov

using LinearAlgebra, ForwardDiff
using NeuralPDE, Lux, ModelingToolkit

export NeuralLyapunovPDESystem, NumericalNeuralLyapunovFunctions

function NeuralLyapunovPDESystem(dynamics, lb, ub, output_dim=1; δ=0.01, relu=(t)->max(0.0,t), fixed_point=nothing)
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

    "Symbolic gradient with respect to (state1, ..., staten)"
    grad(f) = Symbolics.gradient(f, state)

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
    V̇_sym(x) = dynamics(x) ⋅ grad(V_sym(x))
    "V̇ should be negative"
    eq = relu(V̇_sym(state)) ~ 0.0

    # Construct PDESystem
    "V should be 0 at the fixed point"
    bcs = [ V_sym(fixed_point) ~ 0.0 ]
    @named lyapunov_pde_system = PDESystem(eq, bcs, domains, state, u(state))

    # Make Lyapunov function 
    "Numerical form of neural network output"
    u_func(phi, res, x) = vcat([ phi[i](x, res.u.depvar[net_syms[i]]) for i in 1:output_dim ]...)

    "Numerical form of Lyapunov function"
    function V_func(phi, res, x) 
        u_vec = u_func(phi, res, x) .- u_func(phi, res, fixed_point)
        u2 = mapslices(norm, u_vec, dims=[1]).^2
        l = δ*log.(1.0 .+ mapslices(norm, x, dims=[1]).^2)
        u2 .+ l
    end

    return lyapunov_pde_system, V_func
end

function NumericalNeuralLyapunovFunctions(phi, result, lyapunov_func, dynamics; grad=ForwardDiff.gradient)
    "Numerical form of Lyapunov function"
    V_func(state::Matrix) = lyapunov_func(phi, result, state)
    V_func(state::Vector) = first(lyapunov_func(phi, result, state))

    "Numerical gradient of Lyapunov function"
    ∇V_func(state) = mapslices(x -> grad(y -> V_func(y), x), state, dims=[1])

    "Numerical time derivative of Lyapunov function"
    V̇_func(state::Vector) = dynamics(state) ⋅ ∇V_func(state)
    V̇_func(state::Matrix) = reshape(map(x->x[1]⋅x[2], zip(eachslice(dynamics(state), dims=2), eachslice(∇V_func(state), dims=2))), (1,:))

    return V_func, V̇_func
end

end