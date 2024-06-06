using NeuralPDE, Lux, ModelingToolkit
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using NeuralLyapunov
using Random
using Test

Random.seed!(200)

println("Damped Pendulum")

######################### Define dynamics and domain ##########################

@parameters ζ ω_0
defaults = Dict([ζ => 0.5, ω_0 => 1.0])

@variables t θ(t)
Dt = Differential(t)
DDt = Dt^2

eqs = [DDt(θ) + 2ζ * ω_0 * Dt(θ) + ω_0^2 * sin(θ) ~ 0.0]

@named dynamics = ODESystem(
    eqs,
    t,
    [θ],
    [ζ, ω_0];
    defaults = defaults
)

dynamics = structural_simplify(dynamics)
bounds = [
    θ ∈ (-π, π),
    Dt(θ) ∈ (-10.0, 10.0)
]
lb = [-π, -10.0];
ub = [π, 10.0];
p = [defaults[param] for param in parameters(dynamics)]

####################### Specify neural Lyapunov problem #######################

# Define neural network discretization
# We use an input layer that is periodic with period 2π with respect to θ
dim_state = length(bounds)
dim_hidden = 15
dim_output = 2
chain = [Lux.Chain(
             PeriodicEmbedding([1], [2π]),
             Dense(3, dim_hidden, tanh),
             Dense(dim_hidden, dim_hidden, tanh),
             Dense(dim_hidden, 1, use_bias = false)
         ) for _ in 1:dim_output]

# Define neural network discretization
strategy = GridTraining(0.1)
discretization = PhysicsInformedNN(chain, strategy)

# Define neural Lyapunov structure
structure = PositiveSemiDefiniteStructure(
    dim_output;
    pos_def = function (state, fixed_point)
        θ, ω = state
        θ_eq, ω_eq = fixed_point
        log(1.0 + (sin(θ) - sin(θ_eq))^2 + (cos(θ) - cos(θ_eq))^2 + (ω - ω_eq)^2)
    end
)
minimization_condition = DontCheckNonnegativity(check_fixed_point = false)

# Define Lyapunov decrease condition
κ = 20.0
decrease_condition = AsymptoticDecrease(
    strict = true,
    rectifier = (t) -> log(1.0 + exp(κ * t)) / κ
)

# Construct neural Lyapunov specification
spec = NeuralLyapunovSpecification(
    structure,
    minimization_condition,
    decrease_condition
)

############################# Construct PDESystem #############################

@named pde_system = NeuralLyapunovPDESystem(
    ODEFunction(dynamics),
    lb,
    ub,
    spec;
    p = p
)

######################## Construct OptimizationProblem ########################

sym_prob = symbolic_discretize(pde_system, discretization)
prob = discretize(pde_system, discretization)

########################## Solve OptimizationProblem ##########################

res = Optimization.solve(prob, OptimizationOptimisers.Adam(); maxiters = 300)
prob = Optimization.remake(prob, u0 = res.u)
res = Optimization.solve(prob, OptimizationOptimJL.BFGS(); maxiters = 300)

###################### Get numerical numerical functions ######################

V_func, V̇_func = get_numerical_lyapunov_function(
    discretization.phi,
    res.u.depvar,
    structure,
    ODEFunction(dynamics),
    zeros(length(bounds));
    p = p
)

################################## Simulate ###################################

xs = (2 * lb[1]):0.02:(2 * ub[1])
ys = lb[2]:0.02:ub[2]
states = Iterators.map(collect, Iterators.product(xs, ys))
V_predict = vec(V_func(hcat(states...)))
dVdt_predict = vec(V̇_func(hcat(states...)))

#################################### Tests ####################################

# Network structure should enforce positive definiteness
@test V_func([0.0, 0.0]) == 0.0
@test min(V_func([0.0, 0.0]), minimum(V_predict)) ≥ 0.0

# Network structure should enforce periodicity in θ
x0 = (ub .- lb) .* rand(2, 100) .+ lb
@test all(isapprox.(V_func(x0), V_func(x0 .+ [2π, 0.0]); rtol = 1e-3))

# Dynamics should result in a fixed point at the origin
@test V̇_func([0.0, 0.0]) == 0.0

# V̇ should be negative almost everywhere
@test sum(dVdt_predict .> 0) / length(dVdt_predict) < 1e-3

#=
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

p1 = plot(
    xs/pi,
    ys,
    V_predict,
    linetype =
    :contourf,
    title = "V",
    xlabel = "θ/π",
    ylabel = "ω",
    c = :bone_1
    );
p1 = scatter!([-2*pi, 0, 2*pi]/pi, [0, 0, 0], label = "Stable Equilibria", color=:green, markershape=:+);
p1 = scatter!([-pi, pi]/pi, [0, 0], label = "Unstable Equilibria", color=:red, markershape=:x);
p2 = plot(
    xs/pi,
    ys,
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
    xs/pi,
    ys,
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
