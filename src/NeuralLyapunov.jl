module NeuralLyapunov

import ForwardDiff
using ModelingToolkit
using LinearAlgebra
using Optimization
import OptimizationOptimJL

export NeuralLyapunovPDESystem, NumericalNeuralLyapunovFunctions, get_RoA_estimate

function NeuralLyapunovPDESystem(dynamics::Function, lb, ub, output_dim::Integer=1; δ::Real=0.01, ϵ::Real=0.01, relu=(t)->max(0.0,t), fixed_point=nothing)::Tuple{PDESystem, Function}
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
    # u(x) is the symbolic form of neural network output
    u(x) = Num.([ui(x...) for ui in net])
    fixed_point = isnothing(fixed_point) ? zeros(state_dim) : fixed_point
    # V_sym(x) is the symobolic form of the Lyapunov function
    V_sym(x) = ( u(x) - u(fixed_point) ) ⋅ ( u(x) - u(fixed_point) ) + δ*log( 1. + ( x - fixed_point ) ⋅ ( x - fixed_point ) )

    # Define dynamics and Lyapunov conditions
    # V̇_sym(x) is the symbolic time derivative of the Lyapunov function
    V̇_sym(x) = dynamics(x) ⋅ Symbolics.gradient(V_sym(x), x)
    # V̇ should be negative when V < 1, and try not to let V >> 1
    eqs = [ relu(V̇_sym(state)+ϵ*(state - fixed_point)⋅(state - fixed_point))*relu(1 - V_sym(state)) ~ 0.0,
            relu(V_sym(state) - 1) ~ 0.0
            ]
    #eqs = [ relu(V̇_sym(state)+ϵ*(state - fixed_point)⋅(state - fixed_point)) ~ 0.0 ]

    # Construct PDESystem
    bcs = vcat( collect(V_sym(vcat(state[1:i-1], lb[i], state[i+1:end])) ~ 1.1 for i in 1:state_dim),
                collect(V_sym(vcat(state[1:i-1], ub[i], state[i+1:end])) ~ 1.1 for i in 1:state_dim))
    bcs = [ V_sym(fixed_point) ~ 0.0 ] # V should be 0 at the fixed point
    @named lyapunov_pde_system = PDESystem(eqs, bcs, domains, state, u(state))

    # Make Lyapunov function 
    # u_func is the numerical form of neural network output
    u_func(phi, res, x) = reduce( vcat, Array(phi[i](x, res.u.depvar[net_syms[i]])) for i in 1:output_dim )

    # V_func is the numerical form of Lyapunov function
    function V_func(phi, res, x) 
        u_vec = u_func(phi, res, x) .- u_func(phi, res, fixed_point)
        u2 = mapslices(norm, u_vec, dims=[1]).^2
        l = δ*log.(1.0 .+ mapslices(norm, x .- fixed_point, dims=[1]).^2)
        u2 .+ l
    end

    return lyapunov_pde_system, V_func
end

function NeuralLyapunovPDESystem(dynamics::ODEProblem, lb, ub, output_dim::Integer=1; δ::Real=0.01, relu=(t)->max(0.0,t), fixed_point=nothing)::Tuple{PDESystem, Function}
    f = get_dynamics_from_ODEProblem(dynamics)
    return NeuralLyapunovPDESystem(f, lb, ub, output_dim; δ, relu, fixed_point)
end

function NumericalNeuralLyapunovFunctions(phi, result, lyapunov_func, dynamics::Function; grad=ForwardDiff.gradient)
    # Numerical form of Lyapunov function
    V_func(state::AbstractMatrix) = lyapunov_func(phi, result, state)
    V_func(state::AbstractVector) = first(lyapunov_func(phi, result, state))

    # Numerical gradient of Lyapunov function
    ∇V_func(state::AbstractVector) = grad(V_func, state)
    ∇V_func(state::AbstractMatrix) = mapslices(∇V_func, state, dims=[1])

    # Numerical time derivative of Lyapunov function
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
        throw(ErrorException("DAEs are not supported at this time"))
    end
    f(state::AbstractVector) = f_(state)
    f(state::AbstractMatrix) = mapslices(f_, state, dims=[1])
    return f
end

"""
    get_RoA_estimate(V, dVdt, lb, ub; nthreads)

Finds the level of the largest sublevelset in the domain in which the Lyapunov
conditions are met. Specifically finds the largest ρ such that
    V(x) < ρ => lb .< x .< ub && dVdt(x) < 0
To parallelize the search over each face of the bounding box use nthreads to
specify a number of threads.
"""
function get_RoA_estimate(V, dVdt, lb, ub; fixed_point=nothing, ∇V=nothing)

    state_dim = lb isa AbstractArray ? length(lb) : ub isa AbstractArray ? length(ub) : 1
    if !(lb isa AbstractArray)
        lb = ones(state_dim) .* lb
    end
    if !(ub isa AbstractArray)
        ub = ones(state_dim) .* ub
    end

    fixed_point = isnothing(fixed_point) ? zeros(state_dim) : fixed_point

    # Let ρ_max = minimum value of V on the boundary
    # TODO: @view to speed up?
    candidates = Vector{Any}(undef, 2*state_dim)
    for (j, b) in enumerate(vcat(lb, ub))
        i = (j-1) % state_dim + 1
        _lb = vcat(lb[1:i-1], lb[i+1:end])
        _ub = vcat(ub[1:i-1], ub[i+1:end])
        V_boundary = (state, p) -> V(vcat(state[1:i-1], b, state[i:end]))
        ∇V_boundary = if !isnothing(∇V)
            function (state, p) 
                g = ∇V(vcat(state[1:i-1], b, state[i:end]))
                return vcat(g[1:i-1], g[i+1:end])
            end
        else
            nothing
        end
        f = OptimizationFunction(V_boundary, Optimization.AutoForwardDiff(), grad=∇V_boundary)
        state0 = (_lb + _ub)/2
        prob = OptimizationProblem(f, state0, lb = _lb, ub = _ub)
        opt = OptimizationOptimJL.ParticleSwarm(lower = _lb, upper = _ub, n_particles=100)
        res = solve(prob, opt)
        @show candidates[j] = vcat(res.u[1:i-1], b, res.u[i:end])
        @show V(candidates[j])
    end
    @show ρ_max, j_guess = findmin(V, candidates)

    # Find a point just interior of the boundary to start optimization
    guess = candidates[j_guess]
    i_bd = (j_guess-1) % state_dim + 1
    guess[i_bd] = 0.9*(guess[i_bd] - fixed_point[i_bd]) + fixed_point[i_bd]

    # Binary search for max ρ : ( (max V̇(x) : V(x) < ρ) < 0)
    ρ_min = 0.0
    ρ = ρ_max

    function negV̇(dV, x, p)
        dV .= -dVdt(x)
    end
    function negV̇(x, p)
        -dVdt(x)
    end
    function V_param(V_out, x, p)
        V_out .= V(x)
    end
    function V_param(x, p)
        V(x)
    end
    function ∇V_param(∇V_out, x, p)
        ∇V_out .= transpose(∇V(x))
    end
    function ∇V_param(x, p)
        ∇V(x)
    end
    f = OptimizationFunction{true}(negV̇, Optimization.AutoFiniteDiff(); cons=V_param, cons_j=∇V_param)

    while abs(ρ_max - ρ_min) > √eps(Float64)
        # Find max V̇(x) : ρ_min ≤ V(x) < ρ_max
        # Since, we've already verified V(x) < ρ_min and excluded V(x) > ρ_max
        prob = OptimizationProblem{true}(f, guess, lb=lb, ub=ub, lcons=[ρ_min], ucons=[ρ])
        opt = OptimizationOptimJL.IPNewton()
        res = solve(prob, opt, allow_f_increases=true, successive_f_tol=2)
        V̇_max = dVdt(res.u)
        
        if V̇_max > √eps(Float64)
            ρ_max = V(res.u)
            guess = 0.9*(res.u - fixed_point) + fixed_point
        else
            ρ_min = ρ
        end
        ρ = (ρ_max +ρ_min)/2
    end
    return ρ_min
end

end