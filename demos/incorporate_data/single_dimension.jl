using LinearAlgebra
using NeuralPDE, Lux, ModelingToolkit
using Optimization, OptimizationOptimisers, OptimizationOptimJL, NLopt
using Plots
using NeuralLyapunov

######################### Define dynamics and domain ##########################

f(x, p, t) = -x .+ x.^3
lb = [-2.0];
ub = [2.0];
dynamics = ODEFunction(f; syms = [:x])

####################### Specify neural Lyapunov problem #######################

# Define neural network discretization
dim_state = length(lb)
dim_hidden = 15
dim_output = 2
chain = [
    Lux.Chain(
        Dense(dim_state, dim_hidden, tanh),
        Dense(dim_hidden, dim_hidden, tanh),
        Dense(dim_hidden, 1, use_bias = false),
    ) for _ = 1:dim_output
]

# Define neural network discretization
strategy = GridTraining(0.1)
discretization = PhysicsInformedNN(chain, strategy)

# Define neural Lyapunov structure
structure = NonnegativeNeuralLyapunov(
        dim_output; 
        δ = 1e-6
        )
minimization_condition = DontCheckNonnegativity(check_fixed_point = true)

# Define Lyapunov decrease condition
κ = 20.0
decrease_condition_log = AsymptoticDecrease(
    strict = true, 
    relu = (t) -> log(1.0 + exp(κ * t)) / κ
    )

# Construct neural Lyapunov specification
spec_log = NeuralLyapunovSpecification(
    structure,
    minimization_condition,
    decrease_condition_log,
    )

############################# Construct PDESystem #############################

pde_system_log, network_func = NeuralLyapunovPDESystem(
    dynamics,
    lb,
    ub,
    spec_log
)

######################## Construct OptimizationProblem ########################

prob_log = discretize(pde_system_log, discretization)
sym_prob_log = symbolic_discretize(pde_system_log, discretization)

callback = function (p, l)
    println("loss: ", l)
    return false
end

########################## Solve OptimizationProblem ##########################

# Optimize with stricter log version
res = Optimization.solve(prob_log, Adam(); callback = callback, maxiters = 300)

######################### Rebuild OptimizationProblem #########################

println("Switching from log(1 + κ exp(V̇))/κ to max(0,V̇)");

# Set up new decrease condition
decrease_condition_relu = AsymptoticDecrease(strict = true)
spec_relu = NeuralLyapunovSpecification(
    structure,
    minimization_condition,
    decrease_condition_relu,
    )

# Build and discretize new PDESystem
pde_system_relu, _ = NeuralLyapunovPDESystem(dynamics, lb, ub, spec_relu;)
prob_relu = discretize(pde_system_relu, discretization)
sym_prob_relu = symbolic_discretize(pde_system_relu, discretization)

# Rebuild problem with weaker ReLU version
prob_relu = Optimization.remake(prob_relu, u0 = res.u);

######################## Solve new OptimizationProblem ########################

res = Optimization.solve(prob_relu, Adam(); callback = callback, maxiters = 300)
prob_relu = Optimization.remake(prob_relu, u0 = res.u);

println("Switching from Adam to BFGS");
res = Optimization.solve(prob_relu, BFGS(); callback = callback, maxiters = 300)

###################### Get numerical numerical functions ######################
V_func, V̇_func, ∇V_func = NumericalNeuralLyapunovFunctions(
    discretization.phi, 
    res, 
    network_func, 
    structure.V,
    dynamics,
    zeros(length(lb))
    )

################################## Simulate ###################################
xs = lb[1]:0.02:ub[1] 
V_predict = [V_func([x0]) for x0 in xs]
dVdt_predict  = [V̇_func([x0]) for x0 in xs]

# Print statistics
println("V(0.,0.) = ", V_func([0.0]))
println("dVdt(0.,0.) = ", V̇_func([0.0]))
println("V ∋ [", min(V_func([0.0]), minimum(V_predict)), ", ", maximum(V_predict), "]")
println(
    "V̇ ∋ [",
    minimum(dVdt_predict),
    ", ",
    max(V̇_func([0.0]), maximum(dVdt_predict)),
    "]",
)

# Get RoA Estimate
invalid_region = xs[dVdt_predict .>= 0]
invalid_start = maximum(invalid_region[invalid_region .< 0])
invalid_end = minimum(invalid_region[invalid_region .> 0])
valid_region = xs[invalid_start .< xs .< invalid_end]
ρ = min(V_func([first(valid_region)]), V_func([last(valid_region)]))
RoA = valid_region[vec(V_func(transpose(valid_region))) .≤ ρ]

# Plot results
p1 = plot(xs, V_predict, label = "V", xlabel = "x", linewidth=2);
p1 = hline!([ρ], label = "V = $(round(ρ, digits = 4))", legend = :inside)
p1 = vspan!([first(RoA), last(RoA)]; label = "Estimated Region of Attraction", opacity = 0.2, color = :green);
p1 = vspan!([-1, 1]; label = "True Region of Attraction", opacity = 0.2, color = :gray);

p2 = plot(xs, dVdt_predict, label = "dV/dt", xlabel = "x", linewidth=2);
p2 = hline!([0.0], label = "dV/dt = 0", legend = :bottom)
p2 = vspan!([first(RoA), last(RoA)]; label = "Estimated Region of Attraction", opacity = 0.2, color = :green);
p2 = vspan!([-1, 1]; label = "True Region of Attraction", opacity = 0.2, color = :gray);

plot(p1, p2)
