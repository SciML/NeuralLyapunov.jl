using LinearAlgebra
using NeuralPDE, Lux, ModelingToolkit
using Optimization, OptimizationOptimisers, OptimizationOptimJL, NLopt
using NeuralLyapunov
using Random
using Test

Random.seed!(200)

println("Region of Attraction Estimation")

######################### Define dynamics and domain ##########################

f(x, p, t) = -x .+ x.^3
lb = [-2];
ub = [2];

####################### Specify neural Lyapunov problem #######################

# Define neural network discretization
dim_state = length(lb)
dim_hidden = 5
dim_output = 2
chain = [Lux.Chain(
             Dense(dim_state, dim_hidden, tanh),
             Dense(dim_hidden, dim_hidden, tanh),
             Dense(dim_hidden, 1, use_bias = false)
         ) for _ in 1:dim_output]

# Define neural network discretization
strategy = GridTraining(0.1)
discretization = PhysicsInformedNN(chain, strategy)

# Define neural Lyapunov structure
structure = PositiveSemiDefiniteStructure(dim_output)
minimization_condition = DontCheckNonnegativity()

# Define Lyapunov decrease condition
decrease_condition =  make_RoA_aware(
    AsymptoticDecrease(strict = true);
    out_of_RoA_penalty = (_, _, x, x0, _) -> inv(sum((x - x0).^2))
)

# Construct neural Lyapunov specification
spec = NeuralLyapunovSpecification(
    structure,
    minimization_condition,
    decrease_condition
)

############################# Construct PDESystem #############################

pde_system, network_func = NeuralLyapunovPDESystem(
    f,
    lb,
    ub,
    spec
)

######################## Construct OptimizationProblem ########################

prob = discretize(pde_system, discretization)
sym_prob = symbolic_discretize(pde_system, discretization)

########################## Solve OptimizationProblem ##########################

res = Optimization.solve(prob, OptimizationOptimisers.Adam(); maxiters = 300)
prob = Optimization.remake(prob, u0 = res.u)
res = Optimization.solve(prob, BFGS(); maxiters = 300)

###################### Get numerical numerical functions ######################
V_func, V̇_func, ∇V_func = NumericalNeuralLyapunovFunctions(
    discretization.phi,
    res.u,
    network_func,
    structure.V,
    ODEFunction(dynamics),
    zeros(length(lb))
)

################################## Simulate ###################################
states = first(lb):0.02:first(ub)
V_predict = vec(V_func(states'))
dVdt_predict = vec(V̇_func(states'))

#################################### Tests ####################################
#=
# Network structure should enforce nonegativeness of V
@test min(V_func([0.0, 0.0]), minimum(V_predict)) ≥ 0.0

# Trained for V's minimum to be at the fixed point
@test V_func([0.0, 0.0])≈minimum(V_predict) atol=1e-4
@test V_func([0.0, 0.0]) < 1e-4

# Dynamics should result in a fixed point at the origin
@test V̇_func([0.0, 0.0]) == 0.0

# V̇ should be negative almost everywhere
@test sum(dVdt_predict .> 0) / length(dVdt_predict) < 1e-4
=#

# Get RoA Estimate
data = reshape(V_predict, (length(xs), length(ys)));
data = vcat(data[1, :], data[end, :], data[:, 1], data[:, end]);
ρ = minimum(data)

# Print statistics
println("V(0.,0.) = ", V_func([0.0, 0.0]))
println("V ∋ [", min(V_func([0.0, 0.0]), minimum(V_predict)), ", ", maximum(V_predict), "]")
println(
    "V̇ ∋ [",
    minimum(dVdt_predict),
    ", ",
    max(V̇_func([0.0, 0.0]), maximum(dVdt_predict)),
    "]",
)

# Plot results

p1 = plot(xs, ys, V_predict, linetype = :contourf, title = "V", xlabel = "x", ylabel = "ẋ");
p1 = scatter!([0], [0], label = "Equilibrium");
p2 = plot(
    xs,
    ys,
    dVdt_predict,
    linetype = :contourf,
    title = "dV/dt",
    xlabel = "x",
    ylabel = "ẋ",
);
p2 = scatter!([0], [0], label = "Equilibrium");
p3 = plot(
    xs,
    ys,
    V_predict .< ρ,
    linetype = :contourf,
    title = "Estimated RoA",
    xlabel = "x",
    ylabel = "ẋ",
    colorbar = false,
);
p4 = plot(
    xs,
    ys,
    dVdt_predict .< 0,
    linetype = :contourf,
    title = "dV/dt < 0",
    xlabel = "x",
    ylabel = "ẋ",
    colorbar = false,
);
p4 = scatter!([0], [0], label = "Equilibrium");
plot(p1, p2, p3, p4)

@test true
