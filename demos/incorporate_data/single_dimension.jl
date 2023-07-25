using LinearAlgebra
using NeuralPDE, Lux, ModelingToolkit
using Optimization, OptimizationOptimisers, OptimizationOptimJL, NLopt
using Plots
using NeuralLyapunov
using Random

Random.seed!(200)

######################### Define dynamics and domain ##########################

f_true(x, p, t) = -x .+ x.^3
lb = [-2.0];
ub = [2.0];
true_dynamics = ODEFunction(f_true; syms = [:x])

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
decrease_condition = AsymptoticDecrease(strict = true)

# Construct neural Lyapunov specification
spec = NeuralLyapunovSpecification(
    structure,
    minimization_condition,
    decrease_condition,
    )

############################# Construct PDESystem #############################

pde_system, network_func = NeuralLyapunovPDESystem(
    true_dynamics,
    lb,
    ub,
    spec
)

######################## Construct OptimizationProblem ########################

prob = discretize(pde_system, discretization)
sym_prob = symbolic_discretize(pde_system, discretization)

callback = function (p, l)
    println("loss: ", l)
    return false
end

########################## Solve OptimizationProblem ##########################

res = Optimization.solve(prob, Adam(); callback = callback, maxiters = 300)

prob = Optimization.remake(prob, u0 = res.u);
res = Optimization.solve(prob, Adam(); callback = callback, maxiters = 300)

println("Switching from Adam to BFGS");
prob = Optimization.remake(prob, u0 = res.u);
res = Optimization.solve(prob, BFGS(); callback = callback, maxiters = 300)

###################### Get numerical numerical functions ######################
V_func, V̇_func, ∇V_func = NumericalNeuralLyapunovFunctions(
    discretization.phi, 
    res, 
    network_func, 
    structure.V,
    true_dynamics,
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

println("Estimated region of attraction: [$(first(RoA)), $(last(RoA))]")
println("True region of attraction: (-1, 1)")

# Plot results
p1 = plot(xs, V_predict, label = "V", xlabel = "x", linewidth=2);
p1 = hline!([ρ], label = "V = $(round(ρ, digits = 4))", legend = :inside)
p1 = vspan!([first(RoA), last(RoA)]; label = "Estimated Region of Attraction", opacity = 0.2, color = :green);
p1 = vspan!([-1, 1]; label = "True Region of Attraction", color = :gray, fillstyle = :/);

p2 = plot(xs, dVdt_predict, label = "dV/dt", xlabel = "x", linewidth=2);
p2 = hline!([0.0], label = "dV/dt = 0", legend = :bottom)
p2 = vspan!([first(RoA), last(RoA)]; label = "Estimated Region of Attraction", opacity = 0.2, color = :green);
p2 = vspan!([-1, 1]; label = "True Region of Attraction", color = :gray, fillstyle = :/);

plot(p1, p2)

########################### Generate training data ############################
xs = (ub[1] - lb[1]) * rand(20) .+ lb[1]
data = [([x], f_true(x, [], 0.0)) for x in xs]

######################### Define approximate dynamics #########################
Random.seed!(200)
f_approx(x, p, t) = -x
approx_dynamics = ODEFunction(f_approx; syms = [:x])

############################# Construct PDESystem #############################

pde_system, network_func = NeuralLyapunovPDESystem(
    approx_dynamics,
    lb,
    ub,
    spec
)

####################### Specify neural Lyapunov problem #######################
# Specify neural net
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

# Generate addtional loss
data_loss = additional_loss_from_data(data, spec, network_func; fixed_point = [0.0])

# Define neural network discretization
strategy = GridTraining(0.1)
discretization = PhysicsInformedNN(chain, strategy, additional_loss = data_loss)

######################## Construct OptimizationProblem ########################

prob = discretize(pde_system, discretization)
sym_prob = symbolic_discretize(pde_system, discretization)

########################## Solve OptimizationProblem ##########################

res = Optimization.solve(prob, Adam(); callback = callback, maxiters = 300)

prob = Optimization.remake(prob, u0 = res.u);
res = Optimization.solve(prob, Adam(); callback = callback, maxiters = 300)

println("Switching from Adam to BFGS");
prob = Optimization.remake(prob, u0 = res.u);
res = Optimization.solve(prob, BFGS(); callback = callback, maxiters = 300)

###################### Get numerical numerical functions ######################
V_func, V̇_func, ∇V_func = NumericalNeuralLyapunovFunctions(
    discretization.phi, 
    res, 
    network_func, 
    structure.V,
    true_dynamics,
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

println("Estimated region of attraction: [$(first(RoA)), $(last(RoA))]")
println("True region of attraction: (-1, 1)")

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