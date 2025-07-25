using NeuralPDE, Lux, ModelingToolkit, NeuralLyapunov, NeuralLyapunovProblemLibrary
import Boltz.Layers: PeriodicEmbedding
import Optimization, OptimizationOptimisers, OptimizationOptimJL
using StableRNGs, Random
using Test, LinearAlgebra, ForwardDiff

rng = StableRNG(0)
Random.seed!(200)

println("Inverted Pendulum - Policy Search (ODESystem)")

######################### Define dynamics and domain ##########################

p = [0.5, 1.0]
@named driven_pendulum = Pendulum(; driven = true, defaults = p)
t, = independent_variables(driven_pendulum)
θ, τ = unknowns(driven_pendulum)

Dt = Differential(t)
bounds = [
    θ ∈ (0, 2π),
    Dt(θ) ∈ (-2.0, 2.0)
]

upright_equilibrium = [π, 0.0]

####################### Specify neural Lyapunov problem #######################

# Define neural network discretization
# We use an input layer that is periodic with period 2π with respect to θ
dim_state = length(bounds)
dim_hidden = 20
dim_phi = 2
dim_u = 1
dim_output = dim_phi + dim_u
chain = [Chain(
             PeriodicEmbedding([1], [2π]),
             Dense(3, dim_hidden, tanh),
             Dense(dim_hidden, dim_hidden, tanh),
             Dense(dim_hidden, 1)
         ) for _ in 1:dim_output]
ps, st = Lux.setup(rng, chain)

# Define neural network discretization
strategy = QuasiRandomTraining(5000)
discretization = PhysicsInformedNN(chain, strategy; init_params = ps, init_states = st)

# Define neural Lyapunov structure
periodic_pos_def = function (state, fixed_point)
    θ, ω = state
    θ_eq, ω_eq = fixed_point
    return (sin(θ) - sin(θ_eq))^2 + (cos(θ) - cos(θ_eq))^2 + 0.1 * (ω - ω_eq)^2
end

structure = PositiveSemiDefiniteStructure(
    dim_phi;
    pos_def = (x, x0) -> log(1.0 + periodic_pos_def(x, x0))
)
structure = add_policy_search(structure, dim_u)

minimization_condition = DontCheckNonnegativity(check_fixed_point = false)

# Define a periodic Lyapunov decrease condition
decrease_condition = AsymptoticStability(strength = periodic_pos_def)

# Construct neural Lyapunov specification
spec = NeuralLyapunovSpecification(
    structure,
    minimization_condition,
    decrease_condition
)

############################# Construct PDESystem #############################

@named pde_system = NeuralLyapunovPDESystem(
    driven_pendulum,
    bounds,
    spec;
    fixed_point = upright_equilibrium
)

######################## Construct OptimizationProblem ########################

sym_prob = symbolic_discretize(pde_system, discretization)
prob = discretize(pde_system, discretization)

########################## Solve OptimizationProblem ##########################

res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.01); maxiters = 300)
prob = Optimization.remake(prob, u0 = res.u)
res = Optimization.solve(prob, OptimizationOptimJL.BFGS(); maxiters = 300)

###################### Get numerical numerical functions ######################

net = discretization.phi
_θ = res.u.depvar

driven_pendulum_io, _ = structural_simplify(
    driven_pendulum, ([τ], []); simplify = true, split = false)
open_loop_pendulum_dynamics = ODEInputFunction(driven_pendulum_io)
state_order = unknowns(driven_pendulum_io)

V, V̇ = get_numerical_lyapunov_function(
    net,
    _θ,
    structure,
    open_loop_pendulum_dynamics,
    upright_equilibrium;
    p
)

u = get_policy(net, _θ, dim_output, dim_u)

################################## Simulate ###################################

lb = [0.0, -2.0];
ub = [2π, 2.0];
xs = (-2π):0.1:(2π)
ys = lb[2]:0.1:ub[2]
states = Iterators.map(collect, Iterators.product(xs, ys))
V_samples = vec(V(hcat(states...)))
V̇_samples = vec(V̇(hcat(states...)))

#################################### Tests ####################################

# Network structure should enforce positive definiteness
@test V(upright_equilibrium) == 0.0
@test min(V(upright_equilibrium), minimum(V_samples)) ≥ 0.0
@test ForwardDiff.gradient(V, upright_equilibrium) == zeros(2)
@test minimum(eigvals(ForwardDiff.hessian(V, upright_equilibrium))) .≥ 0

# Network structure should enforce periodicity in θ
x0 = (ub .- lb) .* rand(rng, 2, 100) .+ lb
@test maximum(abs, V(x0 .+ [2π, 0.0]) .- V(x0)) < 1e-3

# Training should result in a locally stable fixed point at the upright equilibrium
@test maximum(
    abs,
    open_loop_pendulum_dynamics(upright_equilibrium, u(upright_equilibrium), p, 0.0)
) < 2.5e-3
@test maximum(
    eigvals(
    ForwardDiff.jacobian(
    x -> open_loop_pendulum_dynamics(x, u(x), p, 0.0),
    upright_equilibrium
)
)
) ≤ 0

# Check for local negative definiteness of V̇
@test V̇(upright_equilibrium) == 0.0
@test maximum(abs, ForwardDiff.gradient(V̇, upright_equilibrium)) < 2.5e-3
@test_broken maximum(eigvals(ForwardDiff.hessian(V̇, upright_equilibrium))) ≤ 0

# V̇ should be negative almost everywhere
@test sum(V̇_samples .> 0) / length(V_samples) < 0.01

################################## Simulate ###################################

using OrdinaryDiffEq

state_order = map(st -> SymbolicUtils.isterm(st) ? operation(st) : st, state_order)
state_syms = Symbol.(state_order)

closed_loop_dynamics = ODEFunction(
    (x, p, t) -> open_loop_pendulum_dynamics(x, u(x), p, t);
    sys = SciMLBase.SymbolCache(state_syms, Symbol.(parameters(driven_pendulum)))
)

# Starting still at bottom ...
downward_equilibrium = zeros(2)
ode_prob = ODEProblem(closed_loop_dynamics, downward_equilibrium, [0.0, 120.0], p)
sol = solve(ode_prob, Tsit5())
# plot(sol)

# ...the system should make it to the top
θ_end, ω_end = sol.u[end]
x_end, y_end = sin(θ_end), -cos(θ_end)
@test maximum(abs, [x_end, y_end, ω_end] .- [0.0, 1.0, 0.0]) < 5e-3

# Starting at a random point ...
x0 = lb .+ rand(rng, 2) .* (ub .- lb)
ode_prob = ODEProblem(closed_loop_dynamics, x0, [0.0, 150.0], p)
sol = solve(ode_prob, Tsit5())
# plot(sol)

# ...the system should make it to the top
θ_end, ω_end = sol.u[end]
x_end, y_end = sin(θ_end), -cos(θ_end)
@test maximum(abs, [x_end, y_end, ω_end] .- [0.0, 1.0, 0.0]) < 5e-3

#=
# Print statistics
println("V(π, 0) = ", V(upright_equilibrium))
println(
    "f([π, 0], u([π, 0])) = ",
    open_loop_pendulum_dynamics(upright_equilibrium, u(upright_equilibrium), p, 0.0)
)
println(
    "V ∋ [",
    min(V(upright_equilibrium),
    minimum(V_samples)),
    ", ",
    maximum(V_samples),
    "]"
)
println(
    "V̇ ∋ [",
    minimum(V̇_samples),
    ", ",
    max(V̇(upright_equilibrium), maximum(V̇_samples)),
    "]"
)

# Plot results
using Plots

p1 = plot(
    xs / pi,
    ys,
    V_samples,
    linetype =
    :contourf,
    title = "V",
    xlabel = "θ/π",
    ylabel = "ω",
    c = :bone_1
);
p1 = scatter!([-2 * pi, 0, 2 * pi] / pi, [0, 0, 0],
    label = "Downward Equilibria", color = :red, markershape = :x);
p1 = scatter!(
    [-pi, pi] / pi, [0, 0], label = "Upward Equilibria", color = :green, markershape = :+);
p2 = plot(
    xs / pi,
    ys,
    V̇_samples,
    linetype = :contourf,
    title = "dV/dt",
    xlabel = "θ/π",
    ylabel = "ω",
    c = :binary
);
p2 = scatter!([-2 * pi, 0, 2 * pi] / pi, [0, 0, 0],
    label = "Downward Equilibria", color = :red, markershape = :x);
p2 = scatter!([-pi, pi] / pi, [0, 0], label = "Upward Equilibria", color = :green,
    markershape = :+, legend = false);
p3 = plot(
    xs / pi,
    ys,
    V̇_samples .< 0,
    linetype = :contourf,
    title = "dV/dt < 0",
    xlabel = "θ/π",
    ylabel = "ω",
    colorbar = false,
    linewidth = 0
);
p3 = scatter!([-2 * pi, 0, 2 * pi] / pi, [0, 0, 0],
    label = "Downward Equilibria", color = :red, markershape = :x);
p3 = scatter!([-pi, pi] / pi, [0, 0], label = "Upward Equilibria",
    color = :green, markershape = :+, legend = false);
plot(p1, p2, p3)
=#
