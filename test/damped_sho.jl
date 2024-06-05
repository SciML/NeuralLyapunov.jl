using LinearAlgebra
using NeuralPDE, Lux, ModelingToolkit
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using NeuralLyapunov
using Random
using Test

Random.seed!(200)

println("Damped Simple Harmonic Oscillator")

######################### Define dynamics and domain ##########################

"Simple Harmonic Oscillator Dynamics"
function f(state, p, t)
    ζ, ω_0 = p
    pos = state[1]
    vel = state[2]
    vcat(vel, -2ζ * ω_0 * vel - ω_0^2 * pos)
end
lb = [-5.0, -2.0];
ub = [5.0, 2.0];
p = [0.5, 1.0];
fixed_point = [0.0, 0.0];
dynamics = ODEFunction(f; sys = SciMLBase.SymbolCache([:x, :v], [:ζ, :ω_0]))

####################### Specify neural Lyapunov problem #######################

# Define neural network discretization
dim_state = length(lb)
dim_hidden = 20
dim_output = 5
chain = [Lux.Chain(
             Dense(dim_state, dim_hidden, tanh),
             Dense(dim_hidden, dim_hidden, tanh),
             Dense(dim_hidden, 1)
         ) for _ in 1:dim_output]

# Define training strategy
strategy = GridTraining(0.05)
discretization = PhysicsInformedNN(chain, strategy)

# Define neural Lyapunov structure
structure = NonnegativeNeuralLyapunov(
    dim_output;
    δ = 1e-6
)
minimization_condition = DontCheckNonnegativity(check_fixed_point = true)

# Define Lyapunov decrease condition
# Damped SHO has exponential decrease at a rate of k = ζ * ω_0, so we train to certify that
decrease_condition = ExponentialDecrease(prod(p))

# Construct neural Lyapunov specification
spec = NeuralLyapunovSpecification(
    structure,
    minimization_condition,
    decrease_condition
)

############################# Construct PDESystem #############################

@named pde_system = NeuralLyapunovPDESystem(
    dynamics,
    lb,
    ub,
    spec;
    p = p
)

######################## Construct OptimizationProblem ########################

prob = discretize(pde_system, discretization)
sym_prob = symbolic_discretize(pde_system, discretization)

########################## Solve OptimizationProblem ##########################

res = Optimization.solve(prob, OptimizationOptimisers.Adam(); maxiters = 500)
prob = Optimization.remake(prob, u0 = res.u)
res = Optimization.solve(prob, OptimizationOptimJL.BFGS(); maxiters = 500)

###################### Get numerical numerical functions ######################
V, V̇ = get_numerical_lyapunov_function(
    discretization.phi,
    res.u.depvar,
    structure,
    f,
    fixed_point;
    p = p,
    use_V̇_structure = true
)

################################## Simulate ###################################
xs, ys = [lb[i]:0.02:ub[i] for i in eachindex(lb)]
states = Iterators.map(collect, Iterators.product(xs, ys))
V_samples = vec(V(hcat(states...)))
V̇_samples = vec(V̇(hcat(states...)))

#################################### Tests ####################################

# Network structure should enforce nonegativeness of V
@test min(V(fixed_point), minimum(V_samples)) ≥ 0.0

# Trained for V's minimum to be at the fixed point
@test V(fixed_point)≈minimum(V_samples) atol=1e-4
@test V(fixed_point) < 1e-4

# Dynamics should result in a fixed point at the origin
@test V̇(fixed_point) == 0.0

# V̇ should be negative almost everywhere
@test sum(V̇_samples .> 0) / length(V̇_samples) < 1e-5

#=
# Get RoA Estimate
data = reshape(V_samples, (length(xs), length(ys)));
data = vcat(data[1, :], data[end, :], data[:, 1], data[:, end]);
ρ = minimum(data)

# Print statistics
println("V(0.,0.) = ", V(fixed_point))
println("V ∋ [", min(V(fixed_point), minimum(V_samples)), ", ", maximum(V_samples), "]")
println(
    "V̇ ∋ [",
    minimum(V̇_samples),
    ", ",
    max(V̇(fixed_point), maximum(V̇_samples)),
    "]",
)

# Plot results

p1 = plot(xs, ys, V_samples, linetype = :contourf, title = "V", xlabel = "x", ylabel = "ẋ");
p1 = scatter!([0], [0], label = "Equilibrium");
p2 = plot(
    xs,
    ys,
    V̇_samples,
    linetype = :contourf,
    title = "dV/dt",
    xlabel = "x",
    ylabel = "ẋ",
);
p2 = scatter!([0], [0], label = "Equilibrium");
p3 = plot(
    xs,
    ys,
    V_samples .< ρ,
    linetype = :contourf,
    title = "Estimated RoA",
    xlabel = "x",
    ylabel = "ẋ",
    colorbar = false,
);
p4 = plot(
    xs,
    ys,
    V̇_samples .< 0,
    linetype = :contourf,
    title = "dV/dt < 0",
    xlabel = "x",
    ylabel = "ẋ",
    colorbar = false,
);
p4 = scatter!([0], [0], label = "Equilibrium");
plot(p1, p2, p3, p4)
=#
