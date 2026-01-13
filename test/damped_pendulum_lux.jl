using NeuralPDE, Lux, ModelingToolkit, NeuralLyapunov, NeuralLyapunovProblemLibrary
import Boltz.Layers: PeriodicEmbedding, MLP
import Optimization
using OptimizationOptimisers: Adam
using OptimizationOptimJL: BFGS
using StableRNGs, Random
using Test, LinearAlgebra, ForwardDiff

rng = StableRNG(0)
Random.seed!(200)

println("Damped Pendulum - AdditiveLyapunovNet structure")

######################### Define dynamics and domain ##########################
p = Float32[0.5, 1.0]

@named dynamics = Pendulum(; driven = false, defaults = p)
dynamics = structural_simplify(dynamics)

lb = Float32[-π, -10];
ub = Float32[π, 10];
fixed_point = Float32[0.0, 0.0]

####################### Specify neural Lyapunov problem #######################

# Define neural network discretization
# We use a LyapunovNet with an input layer that is periodic with period 2π with
# respect to θ
dim_state = length(lb)
dim_hidden = 15
dim_output = 2
periodic_embedding_layer = PeriodicEmbedding([1], Float32[2π])
_ps, _st = Lux.setup(rng, periodic_embedding_layer)
periodic_embedding(x) = first(periodic_embedding_layer(x, _ps, _st))
fixed_point_embedded = periodic_embedding(fixed_point)
chain = Chain(
    periodic_embedding_layer,
    AdditiveLyapunovNet(
        MLP(dim_state + 1, (dim_hidden, dim_hidden, dim_output), tanh);
        dim_ϕ = dim_output,
        dim_m = dim_state + 1,
        fixed_point = fixed_point_embedded
    )
)
ps, st = Lux.setup(rng, chain)

# Define neural network discretization
strategy = QuasiRandomTraining(1000)
discretization = PhysicsInformedNN(chain, strategy; init_params = ps, init_states = st)

# Define neural Lyapunov structure and minimization condition
structure = NoAdditionalStructure()
minimization_condition = DontCheckNonnegativity(check_fixed_point = false)

# Define Lyapunov decrease condition
decrease_condition = AsymptoticStability(
    strength = (x, x0) -> sum(abs2, periodic_embedding(x) .- periodic_embedding(x0))
)

# Construct neural Lyapunov specification
spec = NeuralLyapunovSpecification(structure, minimization_condition, decrease_condition)

############################# Construct PDESystem #############################

@named pde_system = NeuralLyapunovPDESystem(ODEFunction(dynamics), lb, ub, spec; p)

######################## Construct OptimizationProblem ########################

sym_prob = symbolic_discretize(pde_system, discretization)
prob = discretize(pde_system, discretization)

########################## Solve OptimizationProblem ##########################

res = Optimization.solve(prob, Adam(0.01f0); maxiters = 300)
prob = Optimization.remake(prob, u0 = res.u)
res = Optimization.solve(prob, BFGS(); maxiters = 300)

###################### Get numerical numerical functions ######################

(V, V̇) = get_numerical_lyapunov_function(
    discretization.phi,
    res.u,
    structure,
    ODEFunction(dynamics),
    fixed_point;
    p
)

################################## Simulate ###################################

θs = (2 * lb[1]):0.02f0:(2 * ub[1])
ωs = lb[2]:0.1f0:ub[2]
states = mapreduce(collect, hcat, Iterators.product(θs, ωs))
V_predict = vec(V(states))
dVdt_predict = vec(V̇(states))

#################################### Tests ####################################

# Network structure should enforce positive definiteness
V0 = V(fixed_point)[]
@test V0 == 0.0
@test min(V0, minimum(V_predict)) ≥ 0.0

# Check local positive definiteness at fixed point
@test maximum(abs, ForwardDiff.gradient(first ∘ V, fixed_point)) ≤ 1.0e-6
@test minimum(eigvals(ForwardDiff.hessian(first ∘ V, fixed_point))) .≥ 0

# Network structure should enforce periodicity in θ
x0 = (ub .- lb) .* rand(rng, Float32, 2, 100) .+ lb
@test maximum(abs, V(x0 .+ Float32[2π, 0.0]) .- V(x0)) .≤ 1.0e-3

# Check local negative definiteness at fixed point
@test V̇(fixed_point)[] == 0.0
@test maximum(abs, ForwardDiff.gradient(first ∘ V̇, fixed_point)) ≤ 2.0e-6
@test maximum(eigvals(ForwardDiff.hessian(first ∘ V̇, fixed_point))) ≤ 0

# V̇ should be negative almost everywhere (global negative definiteness)
@test sum(dVdt_predict .> 0) / length(dVdt_predict) < 1.0e-3

#=
# Print statistics
println("V(0.,0.) = ", V(fixed_point)[])
println("V ∋ [", min(V(fixed_point)[], minimum(V_predict)), ", ", maximum(V_predict), "]")
println(
    "V̇ ∋ [",
    minimum(dVdt_predict),
    ", ",
    max(V̇(fixed_point)[], maximum(dVdt_predict)),
    "]",
)

# Plot results
using Plots

p1 = plot(
    θs/pi,
    ωs,
    V_predict,
    linetype = :contourf,
    title = "V",
    xlabel = "θ/π",
    ylabel = "ω",
    c = :bone_1
    );
p1 = scatter!([-2*pi, 0, 2*pi]/pi, [0, 0, 0], label = "Stable Equilibria", color=:green, markershape=:+);
p1 = scatter!([-pi, pi]/pi, [0, 0], label = "Unstable Equilibria", color=:red, markershape=:x);
p2 = plot(
    θs/pi,
    ωs,
    dVdt_predict,
    linetype = :contourf,
    title = "dV/dt",
    xlabel = "θ/π",
    ylabel = "ω",
    c = :binary
);
p2 = scatter!([-2*pi, 0, 2*pi]/pi, [0, 0, 0], label = "Stable Equilibria", color=:green, markershape=:+);
p2 = scatter!([-pi, pi]/pi, [0, 0], label = "Unstable Equilibria", color=:red, markershape=:x, legend=false);
p3 = plot(
    θs/pi,
    ωs,
    dVdt_predict .< 0,
    linetype = :contourf,
    title = "dV/dt < 0",
    xlabel = "θ/π",
    ylabel = "ω",
    colorbar = false,
    linewidth = 0
);
p3 = scatter!([-2*pi, 0, 2*pi]/pi, [0, 0, 0], label = "Stable Equilibria", color=:green, markershape=:+);
p3 = scatter!([-pi, pi]/pi, [0, 0], label = "Unstable Equilibria", color=:red, markershape=:x, legend=false);
plot(p1, p2, p3)
=#
