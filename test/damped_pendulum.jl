using NeuralPDE, Lux, ComponentArrays, ModelingToolkit, NeuralLyapunov
import Boltz.Layers: PeriodicEmbedding, MLP
import Optimization
using OptimizationOptimisers: Adam
using OptimizationOptimJL: BFGS
using StableRNGs, Random
using Test, LinearAlgebra, ForwardDiff

rng = StableRNG(0)
Random.seed!(200)

println("Damped Pendulum")

######################### Define dynamics and domain ##########################

@parameters ζ ω_0
defaults = Dict([ζ => 0.5, ω_0 => 1.0])

@independent_variables t
@variables θ(t)
Dt = Differential(t)
DDt = Dt^2

eqs = [DDt(θ) + 2ζ * ω_0 * Dt(θ) + ω_0^2 * sin(θ) ~ 0.0]

@named dynamics = ODESystem(eqs, t, [θ], [ζ, ω_0]; defaults)

dynamics = structural_simplify(dynamics)
bounds = [
    θ ∈ (-π, π),
    Dt(θ) ∈ (-10.0, 10.0),
]
lb = [-π, -10.0];
ub = [π, 10.0];
p = [defaults[param] for param in parameters(dynamics)]
fixed_point = [0.0, 0.0]

####################### Specify neural Lyapunov problem #######################

# Define neural network discretization
# We use an input layer that is periodic with period 2π with respect to θ
dim_state = length(bounds)
dim_hidden = 15
dim_output = 2
chain = [
    Chain(
            PeriodicEmbedding([1], [2π]),
            MLP(dim_state + 1, (dim_hidden, dim_hidden, 1), tanh)
        ) for _ in 1:dim_output
]
ps, st = Lux.setup(rng, chain)
ps = ps |> ComponentArray |> f64
st = st |> f64

# Define neural network discretization
strategy = QuasiRandomTraining(2000)
discretization = PhysicsInformedNN(chain, strategy; init_params = ps, init_states = st)

# Define neural Lyapunov structure and corresponding minimization condition
periodic_pos_def = function (state, fixed_point)
    θ, ω = state
    θ_eq, ω_eq = fixed_point
    return (sin(θ) - sin(θ_eq))^2 + (cos(θ) - cos(θ_eq))^2 + (ω - ω_eq)^2
end
structure = PositiveSemiDefiniteStructure(
    dim_output;
    pos_def = (x, x0) -> log(1.0 + periodic_pos_def(x, x0))
)
minimization_condition = DontCheckNonnegativity(check_fixed_point = false)

# Define Lyapunov decrease condition
decrease_condition = AsymptoticStability(
    strength = periodic_pos_def,
    rectifier = (t) -> log(one(t) + exp(t))
)

# Construct neural Lyapunov specification
spec = NeuralLyapunovSpecification(structure, minimization_condition, decrease_condition)

############################# Construct PDESystem #############################

@named pde_system = NeuralLyapunovPDESystem(ODEFunction(dynamics), lb, ub, spec; p)

######################## Construct OptimizationProblem ########################

sym_prob = symbolic_discretize(pde_system, discretization)
prob = discretize(pde_system, discretization)

########################## Solve OptimizationProblem ##########################

res = Optimization.solve(prob, Adam(0.1); maxiters = 500)
prob = Optimization.remake(prob, u0 = res.u)
res = Optimization.solve(prob, BFGS(); maxiters = 500)

###################### Get numerical numerical functions ######################

(V, V̇) = get_numerical_lyapunov_function(
    discretization.phi,
    res.u.depvar,
    structure,
    ODEFunction(dynamics),
    zeros(length(bounds));
    p
)

################################## Simulate ###################################

θs = (2 * lb[1]):0.02:(2 * ub[1])
ωs = lb[2]:0.02:ub[2]
states = mapreduce(collect, hcat, Iterators.product(θs, ωs))
V_predict = vec(V(states))
dVdt_predict = vec(V̇(states))

#################################### Tests ####################################

# Network structure should enforce positive definiteness
@test V(fixed_point) == 0.0
@test min(V(fixed_point), minimum(V_predict)) ≥ 0.0

# Check local positive definiteness at fixed point
@test ForwardDiff.gradient(V, fixed_point) == zeros(2)
@test minimum(eigvals(ForwardDiff.hessian(V, fixed_point))) .≥ 0

# Network structure should enforce periodicity in θ
x0 = (ub .- lb) .* rand(rng, 2, 100) .+ lb
@test maximum(abs, V(x0 .+ [2π, 0.0]) .- V(x0)) .≤ 1.0e-3

# Check local negative definiteness at fixed point
@test V̇(fixed_point) == 0.0
@test ForwardDiff.gradient(V̇, fixed_point) == zeros(2)
@test maximum(eigvals(ForwardDiff.hessian(V̇, fixed_point))) ≤ 0

# V̇ should be negative almost everywhere (global negative definiteness)
@test sum(dVdt_predict .> 0) / length(dVdt_predict) < 1.0e-3

#=
# Print statistics
println("V(0.,0.) = ", V(fixed_point))
println("V ∋ [", min(V(fixed_point), minimum(V_predict)), ", ", maximum(V_predict), "]")
println(
    "V̇ ∋ [",
    minimum(dVdt_predict),
    ", ",
    max(V̇(fixed_point), maximum(dVdt_predict)),
    "]",
)

# Plot results

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
