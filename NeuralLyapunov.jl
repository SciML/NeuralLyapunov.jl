using LinearAlgebra, ForwardDiff
using Optimization, OptimizationOptimisers, OptimizationOptimJL, NLopt
using Plots
using NeuralPDE, Lux, ModelingToolkit

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

# Set up SHO system
function SHO_dynamics(state) 
    pos = transpose(state[1,:]); vel = transpose(state[2,:])
    [vel; -vel-pos]
end
lb = [-2*pi, -10.0]; ub = [2*pi, 10.0]

# Make log version
output_dim = 1
κ=20.0
pde_system_log, lyapunov_func = NeuralLyapunovPDESystem(SHO_dynamics, lb, ub, output_dim, relu=(t)->log(1.0 + exp( κ * t)))

# Set up neural net 
state_dim = length(lb)
dim_hidden = 15
chain = [Lux.Chain(
                Dense(state_dim, dim_hidden, tanh), 
                Dense(dim_hidden, dim_hidden, tanh),
                Dense(dim_hidden, 1, use_bias=false)
                )
            for _ in 1:output_dim
            ]

# Define neural network discretization
strategy = GridTraining(0.1)
discretization = PhysicsInformedNN(chain, strategy)

# Build optimization problem
prob_log = discretize(pde_system_log, discretization)

callback = function (p, l)
    println("loss: ", l)
    return false
end

# Optimize log version
res = Optimization.solve(prob_log, Adam(); callback=callback, maxiters=300)

# Optimize ReLU verion
pde_system_relu, _ = NeuralLyapunovPDESystem(SHO_dynamics, lb, ub, output_dim)
prob_relu = discretize(pde_system_relu, discretization)
prob_relu = Optimization.remake(prob_relu, u0=res.u); println("Switching from log(1 + κ exp(V̇)) to max(0,V̇)")
res = Optimization.solve(prob_relu, Adam(); callback=callback, maxiters=300)
prob_relu = Optimization.remake(prob_relu, u0=res.u); println("Switching from Adam to BFGS")
res = Optimization.solve(prob_relu, BFGS(); callback=callback, maxiters=300)

# Get numerical numerical functions
V_func, V̇_func = NumericalNeuralLyapunovFunctions(discretization.phi, res, lyapunov_func, SHO_dynamics)

# Simulate
xs,ys = [lb[i]:0.02:ub[i] for i in eachindex(lb)]
states = Iterators.map(x->[x...], Iterators.product(ys, xs))
V_predict = V_func(hcat(states...))
dVdt_predict = V̇_func(hcat(states...))

# Print statistics
println("V(0.,0.) = ", V_func([0.,0.]))
println("V ∋ [", min(V_func([0.,0.]), minimum(V_predict)), ", ", maximum(V_predict), "]")
println("V̇ ∋ [", minimum(dVdt_predict), ", ", max(V̇_func([0.,0.]), maximum(dVdt_predict)), "]")

# Plot results

p1 = plot(xs, ys, reshape(V_predict, (length(ys), length(xs))), linetype=:contourf, title = "V", xlabel="x", ylabel="ẋ");
p2 = plot(xs, ys, reshape(dVdt_predict, (length(ys), length(xs))), linetype=:contourf, title="dV/dt", xlabel="x", ylabel="ẋ");
#p2 = scatter!([-pi, pi], [0., 0.], label="Unstable equilibria");
#p2 = scatter!([-2*pi, 0., 2*pi], [0., 0., 0.], label="Stable equilibria");
plot(p1, p2)
# savefig("Lyapunov_sol")