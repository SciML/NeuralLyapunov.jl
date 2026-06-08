using NeuralPDE, Lux, NeuralLyapunov, ComponentArrays
using Boltz.Layers: MLP
import Optimization
using OptimizationOptimisers: Adam
using OptimizationOptimJL: BFGS
using StableRNGs, Random
using Test, LinearAlgebra, ForwardDiff

rng = StableRNG(0)
Random.seed!(200)

println("Damped Simple Harmonic Oscillator")

######################### Define dynamics and domain ##########################

"Simple Harmonic Oscillator Dynamics"
function f(state, p, t)
    ζ, ω_0 = p
    pos = state[1]
    vel = state[2]
    return vcat(vel, -2ζ * ω_0 * vel - ω_0^2 * pos)
end
lb = [-5.0, -2.0];
ub = [5.0, 2.0];
p = [0.5, 1.0];
fixed_point = [0.0, 0.0];
dynamics = ODEFunction(f; sys = SciMLBase.SymbolCache([:x, :v], [:ζ, :ω_0]))

####################### Specify neural Lyapunov problem #######################

# Define neural network discretization
dim_state = length(lb)
dim_hidden = 10
dim_output = 4
chain = [MLP(dim_state, (dim_hidden, dim_hidden, 1), tanh) for _ in 1:dim_output]
ps, st = Lux.setup(rng, chain)
ps = ps |> ComponentArray |> f64
st = st |> f64

# Define training strategy
strategy = QuasiRandomTraining(1000)
discretization = PhysicsInformedNN(chain, strategy; init_params = ps, init_states = st)

# Define neural Lyapunov structure and corresponding minimization condition
structure = NonnegativeStructure(dim_output; δ = 5.0)
minimization_condition = DontCheckNonnegativity(check_fixed_point = true)

# Define Lyapunov decrease condition
# Damped SHO has exponential decrease at a rate of k = ζ * ω_0, so we train to certify that
decrease_condition = ExponentialStability(prod(p))

# Construct neural Lyapunov specification
spec = NeuralLyapunovSpecification(structure, minimization_condition, decrease_condition)

############################# Construct PDESystem #############################

@named pde_system = NeuralLyapunovPDESystem(dynamics, lb, ub, spec; p)

######################## Construct OptimizationProblem ########################

prob = discretize(pde_system, discretization)
sym_prob = symbolic_discretize(pde_system, discretization)

########################## Solve OptimizationProblem ##########################

res = Optimization.solve(prob, Adam(); maxiters = 450)
prob = Optimization.remake(prob, u0 = res.u)
res = Optimization.solve(prob, BFGS(); maxiters = 300)

###################### Get numerical numerical functions ######################
(V, V̇) = get_numerical_lyapunov_function(
    discretization.phi,
    res.u.depvar,
    structure,
    f,
    fixed_point;
    p,
    use_V̇_structure = true
)

################################## Simulate ###################################
Δx = (ub[1] - lb[1]) / 100
Δv = (ub[2] - lb[2]) / 100
xs = lb[1]:Δx:ub[1]
vs = lb[2]:Δv:ub[2]
states = Iterators.map(collect, Iterators.product(xs, vs))
V_samples = vec(V(reduce(hcat, states)))
V̇_samples = vec(V̇(reduce(hcat, states)))

#################################### Tests ####################################

# Network structure should enforce nonegativeness of V
V_min, i_min = findmin(V_samples)
state_min = collect(states)[i_min]
V_min, state_min = if V(fixed_point) ≤ V_min
    V(fixed_point), fixed_point
else
    V_min, state_min
end
@test V_min ≥ 0.0

# Trained for V's minimum to be near the fixed point
@test all(abs.(state_min .- fixed_point) .≤ [Δx, Δv])

# Check local negative semidefiniteness of V̇ at fixed point
@test V̇(fixed_point) == 0.0
# @test ForwardDiff.gradient(V̇, fixed_point) == zeros(2)
# @test maximum(eigvals(ForwardDiff.hessian(V̇, fixed_point))) ≤ 0.0

# V̇ should be negative almost everywhere
@test sum(V̇_samples .> 0) / length(V̇_samples) < 1.0e-3

#=
# Print statistics
println("V(0.,0.) = ", V(fixed_point))
println("V ∋ [", V_min, ", ", maximum(V_samples), "]")
println("Minimal sample of V is at ", state_min)
println(
    "V̇ ∋ [",
    minimum(V̇_samples),
    ", ",
    max(V̇(fixed_point), maximum(V̇_samples)),
    "]",
)

# Plot results
using Plots

p1 = plot(xs, vs, V_samples, linetype = :contourf, title = "V", xlabel = "x", ylabel = "ẋ");
p1 = scatter!([0], [0], label = "Equilibrium");
p2 = plot(
    xs,
    vs,
    V̇_samples,
    linetype = :contourf,
    title = "dV/dt",
    xlabel = "x",
    ylabel = "ẋ",
);
p2 = scatter!([0], [0], label = "Equilibrium");
plot(p1, p2)
=#
