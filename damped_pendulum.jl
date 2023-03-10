using LinearAlgebra
using NeuralPDE, Lux, ModelingToolkit
using Optimization, OptimizationOptimisers, OptimizationOptimJL, NLopt
using Plots
if !@isdefined(NeuralLyapunov) # Since it's not a normal package, we do this
    include("./NeuralLyapunov.jl")
end
using .NeuralLyapunov

# Define dynamics
"Pendulum Dynamics"
function pendulum_dynamics(state::AbstractMatrix{T})::AbstractMatrix{T} where T <:Number
    pos = transpose(state[1,:]); vel = transpose(state[2,:])
    vcat(vel, -vel-sin.(pos))
end
function pendulum_dynamics(state::AbstractVector{T})::AbstractVector{T} where T <:Number
    pos = state[1]; vel = state[2]
    vcat(vel, -vel-sin.(pos))
end
lb = [-2*pi, -10.0]; ub = [2*pi, 10.0]

# Make log version
dim_output = 2
κ=20.0
pde_system_log, lyapunov_func = NeuralLyapunovPDESystem(pendulum_dynamics, lb, ub, dim_output, relu=(t)->log(1.0 + exp( κ * t)))

# Define neural network discretization
dim_state = length(lb)
dim_hidden = 15
chain = [Lux.Chain(
                Dense(dim_state, dim_hidden, tanh), 
                Dense(dim_hidden, dim_hidden, tanh),
                Dense(dim_hidden, 1, use_bias=false)
                )
            for _ in 1:dim_output
            ]

# Define neural network discretization
strategy = GridTraining(0.1)
discretization = PhysicsInformedNN(chain, strategy)

# Build optimization problem
prob_log = discretize(pde_system_log, discretization)
sym_prob_log = symbolic_discretize(pde_system_log, discretization)

callback = function (p, l)
    println("loss: ", l)
    return false
end

# Optimize with stricter log version
res = Optimization.solve(prob_log, Adam(); callback=callback, maxiters=300)

# Rebuild with weaker ReLU version
pde_system_relu, _ = NeuralLyapunovPDESystem(SHO_dynamics, lb, ub, dim_output)
prob_relu = discretize(pde_system_relu, discretization)
sym_prob_relu = symbolic_discretize(pde_system_relu, discretization)
prob_relu = Optimization.remake(prob_relu, u0=res.u); println("Switching from log(1 + κ exp(V̇)) to max(0,V̇)")
res = Optimization.solve(prob_relu, Adam(); callback=callback, maxiters=300)
prob_relu = Optimization.remake(prob_relu, u0=res.u); println("Switching from Adam to BFGS")
res = Optimization.solve(prob_relu, BFGS(); callback=callback, maxiters=300)

# Get numerical numerical functions
V_func, V̇_func = NumericalNeuralLyapunovFunctions(discretization.phi, res, lyapunov_func, pendulum_dynamics)

# Simulate
xs,ys = [lb[i]:0.02:ub[i] for i in eachindex(lb)]
states = Iterators.map(x->[x...], Iterators.product(xs, ys))
V_predict = vec(V_func(hcat(states...)))
dVdt_predict = vec(V̇_func(hcat(states...)))
# V_predict = [V_func([x0,y0]) for y0 in ys for x0 in xs]
# dVdt_predict  = [V̇_func([x0,y0]) for y0 in ys for x0 in xs]

# Print statistics
println("V(0.,0.) = ", V_func([0.,0.]))
println("V ∋ [", min(V_func([0.,0.]), minimum(V_predict)), ", ", maximum(V_predict), "]")
println("V̇ ∋ [", minimum(dVdt_predict), ", ", max(V̇_func([0.,0.]), maximum(dVdt_predict)), "]")

# Plot results

p1 = plot(xs, ys, V_predict, linetype=:contourf, title = "V", xlabel="x", ylabel="ẋ");
p2 = plot(xs, ys, dVdt_predict, linetype=:contourf, title="dV/dt", xlabel="x", ylabel="ẋ");
p2 = scatter!([-pi, pi], [0., 0.], label="Unstable equilibria");
p2 = scatter!([-2*pi, 0., 2*pi], [0., 0., 0.], label="Stable equilibria");
plot(p1, p2)
# savefig("Pendulum")