using LinearAlgebra
using NeuralPDE, Lux, ModelingToolkit
using Optimization, OptimizationOptimisers, OptimizationOptimJL, NLopt
using NeuralLyapunov
using Random
using Test

Random.seed!(200)

println("Region of Attraction Estimation")

######################### Define dynamics and domain ##########################

f(x, p, t) = -x .+ x .^ 3
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
decrease_condition = make_RoA_aware(AsymptoticDecrease(strict = true))

# Construct neural Lyapunov specification
spec = NeuralLyapunovSpecification(
    structure,
    minimization_condition,
    decrease_condition
)

############################# Construct PDESystem #############################

@named pde_system = NeuralLyapunovPDESystem(
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
V_func, V̇_func = NumericalNeuralLyapunovFunctions(
    discretization.phi,
    res.u,
    structure,
    f,
    zeros(length(lb))
)

################################## Simulate ###################################
states = first(lb):0.001:first(ub)
V_predict = vec(V_func(states'))
dVdt_predict = vec(V̇_func(states'))

# Calculated RoA estimate
ρ = decrease_condition.ρ
RoA_states = states[vec(V_func(transpose(states))) .≤ ρ]
RoA = (first(RoA_states), last(RoA_states))

#################################### Tests ####################################

# Network structure should enforce positive definiteness of V
@test min(V_func([0.0]), minimum(V_predict)) ≥ 0.0
@test V_func([0.0]) == 0.0

# Dynamics should result in a fixed point at the origin
@test V̇_func([0.0]) == 0.0

# V̇ should be negative everywhere in the region of attraction except the fixed point
@test all(V̇_func(transpose(RoA_states[RoA_states .!= 0.0])) .< 0)

# The estimated region of attraction should be a subset of the real region of attraction
@test first(RoA) ≥ -1.0 && last(RoA) ≤ 1.0

#=
# Print statistics
println("V(0.,0.) = ", V_func([0.0]))
println("V ∋ [", min(V_func([0.0]), minimum(V_predict)), ", ", maximum(V_predict), "]")
println(
    "V̇ ∋ [",
    minimum(dVdt_predict),
    ", ",
    max(V̇_func([0.0]), maximum(dVdt_predict)),
    "]",
)
println("True region of attraction: (-1, 1)")
println("Estimated region of attraction: ", RoA)

# Plot results
using Plots

p_V = plot(states, V_predict, label = "V", xlabel = "x", linewidth=2);
p_V = hline!([ρ], label = "V = ρ", legend = :top);
p_V = vspan!(collect(RoA); label = "Estimated Region of Attraction", color = :gray, fillstyle = :/);
p_V = vspan!([-1, 1]; label = "True Region of Attraction", opacity = 0.2, color = :green);

p_V̇ = plot(states, dVdt_predict, label = "dV/dt", xlabel = "x", linewidth=2);
p_V̇ = hline!([0.0], label = "dV/dt = 0", legend = :top);
p_V̇ = vspan!(collect(RoA); label = "Estimated Region of Attraction", color = :gray, fillstyle = :/);
p_V̇ = vspan!([-1, 1]; label = "True Region of Attraction", opacity = 0.2, color = :green);

plt = plot(p_V, p_V̇)
=#
