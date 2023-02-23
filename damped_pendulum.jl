using LinearAlgebra
using NeuralPDE, Lux
using Optimization, OptimizationOptimisers, OptimizationOptimJL, NLopt
using Plots
include("./NeuralLyapunov.jl") # Change this if we start using the module in more than one file (like a package)

# Set up SHO system
function pendulum_dynamics(state) 
    pos = transpose(state[1,:]); vel = transpose(state[2,:])
    [vel; -vel-sin.(pos)]
end
lb = [-2*pi, -10.0]; ub = [2*pi, 10.0]

# Make log version
output_dim = 1
κ=20.0
pde_system_log, lyapunov_func = NeuralLyapunov.NeuralLyapunovPDESystem(pendulum_dynamics, lb, ub, output_dim, relu=(t)->log(1.0 + exp( κ * t)))

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
pde_system_relu, _ = NeuralLyapunov.NeuralLyapunovPDESystem(pendulum_dynamics, lb, ub, output_dim)
prob_relu = discretize(pde_system_relu, discretization)
prob_relu = Optimization.remake(prob_relu, u0=res.u); println("Switching from log(1 + κ exp(V̇)) to max(0,V̇)")
res = Optimization.solve(prob_relu, Adam(); callback=callback, maxiters=300)
prob_relu = Optimization.remake(prob_relu, u0=res.u); println("Switching from Adam to BFGS")
res = Optimization.solve(prob_relu, BFGS(); callback=callback, maxiters=300)

# Get numerical numerical functions
V_func, V̇_func = NeuralLyapunov.NumericalNeuralLyapunovFunctions(discretization.phi, res, lyapunov_func, pendulum_dynamics)

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
p2 = scatter!([-pi, pi], [0., 0.], label="Unstable equilibria");
p2 = scatter!([-2*pi, 0., 2*pi], [0., 0., 0.], label="Stable equilibria");
plot(p1, p2)
# savefig("SHO")