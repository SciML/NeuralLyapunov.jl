using NeuralPDE, Lux, NeuralLyapunov
import Boltz.Layers: PeriodicEmbedding, MLP
import Optimization
using OptimizationOptimisers: Adam
using OptimizationOptimJL: BFGS
using StableRNGs, Random
using Test, LinearAlgebra, ForwardDiff

rng = StableRNG(0)
Random.seed!(200)

println("Inverted Pendulum - Policy Search")

######################### Define dynamics and domain ##########################

function open_loop_pendulum_dynamics(x, u, p, t)
    θ, ω = x
    ζ, ω_0 = p
    τ = u[]
    return [ω, -2ζ * ω_0 * ω - ω_0^2 * sin(θ) + τ]
end

lb = Float32[0.0, -2.0];
ub = Float32[2π, 2.0];
upright_equilibrium = Float32[π, 0.0]
p = Float32[0.5, 1.0]
state_syms = [:θ, :ω]
parameter_syms = [:ζ, :ω_0]

####################### Specify neural Lyapunov problem #######################

# Define neural network discretization
# We use an input layer that is periodic with period 2π with respect to θ
dim_state = length(lb)
dim_hidden = 25
dim_phi = 3
dim_u = 1
dim_output = dim_phi + dim_u
chain = [Chain(
             PeriodicEmbedding([1], Float32[2π]),
             MLP(dim_state + 1, (dim_hidden, dim_hidden, 1), tanh)
         ) for _ in 1:dim_output]
ps, st = Lux.setup(rng, chain)

# Define neural network discretization
strategy = QuasiRandomTraining(10000)
discretization = PhysicsInformedNN(chain, strategy; init_params = ps, init_states = st)

# Define neural Lyapunov structure and corresponding minimization condition
periodic_pos_def = function (state, fixed_point)
    θ, ω = state
    θ_eq, ω_eq = fixed_point
    return (sin(θ) - sin(θ_eq))^2 + (cos(θ) - cos(θ_eq))^2 + (ω - ω_eq)^2 / 10
end

structure = PositiveSemiDefiniteStructure(
    dim_phi;
    pos_def = (x, x0) -> log(1 + periodic_pos_def(x, x0))
)
structure = add_policy_search(structure, dim_u)

minimization_condition = DontCheckNonnegativity(check_fixed_point = false)

# Define a periodic Lyapunov decrease condition
decrease_condition = AsymptoticStability(strength = periodic_pos_def)

# Construct neural Lyapunov specification
spec = NeuralLyapunovSpecification(structure, minimization_condition, decrease_condition)

############################# Construct PDESystem #############################

@named pde_system = NeuralLyapunovPDESystem(
    open_loop_pendulum_dynamics,
    lb,
    ub,
    spec;
    fixed_point = upright_equilibrium,
    p,
    state_syms,
    parameter_syms,
    policy_search = true
)

######################## Construct OptimizationProblem ########################

sym_prob = symbolic_discretize(pde_system, discretization)
prob = discretize(pde_system, discretization)

########################## Solve OptimizationProblem ##########################

res = Optimization.solve(prob, Adam(0.05f0); maxiters = 300)
prob = Optimization.remake(prob, u0 = res.u)
res = Optimization.solve(prob, BFGS(); maxiters = 300)

########################### Get numerical functions ###########################

net = discretization.phi
θ = res.u.depvar

(V, V̇) = get_numerical_lyapunov_function(
    net,
    θ,
    structure,
    open_loop_pendulum_dynamics,
    upright_equilibrium;
    p
)

u = get_policy(net, θ, dim_output, dim_u)

closed_loop_pendulum_dynamics(x) = open_loop_pendulum_dynamics(x, u(x), p, 0)

################################## Simulate ###################################

θs = (-2.0f0 * π):0.02f0:(2.0f0 * π)
ωs = lb[2]:0.02f0:ub[2]
states = mapreduce(collect, hcat, Iterators.product(θs, ωs))
V_predict = vec(V(states))
dVdt_predict = vec(V̇(states))

#################################### Tests ####################################

# Network structure should enforce positive definiteness
@test V(upright_equilibrium) == 0.0
@test min(V(upright_equilibrium), minimum(V_predict)) ≥ 0.0
@test ForwardDiff.gradient(V, upright_equilibrium) == zeros(2)
@test minimum(eigvals(ForwardDiff.hessian(V, upright_equilibrium))) ≥ 0

# Network structure should enforce periodicity in θ
x0 = (ub .- lb) .* rand(rng, Float32, 2, 100) .+ lb
@test maximum(abs, V(x0 .+ Float32[2π, 0.0]) .- V(x0)) < 1e-3

# Training should result in a locally stable fixed point at the upright equilibrium
# Check for approximately zero angular acceleration
@test abs(closed_loop_pendulum_dynamics(upright_equilibrium)[2]) < 2.5e-3
# Check for nonpositive eigenvalues of the Jacobian
@test_broken maximum(
    eigvals(
    ForwardDiff.jacobian(closed_loop_pendulum_dynamics, upright_equilibrium)
)
) ≤ 0

# Check for local negative definiteness of V̇
@test V̇(upright_equilibrium) == 0.0
@test maximum(abs, ForwardDiff.gradient(V̇, upright_equilibrium)) < 2.5e-3
@test_broken maximum(eigvals(ForwardDiff.hessian(V̇, upright_equilibrium))) ≤ 0

# V̇ should be negative almost everywhere
@test sum(dVdt_predict .> 0) / length(dVdt_predict) < 5e-3

################################## Simulate ###################################

using OrdinaryDiffEq

closed_loop_dynamics = ODEFunction(
    (x, p, t) -> open_loop_pendulum_dynamics(x, u(x), p, t);
    sys = SciMLBase.SymbolCache(state_syms, parameter_syms)
)

# Starting still at bottom
downward_equilibrium = zeros(Float32, 2)
ode_prob = ODEProblem(closed_loop_dynamics, downward_equilibrium, 75.0f0, p)
sol = solve(ode_prob, Tsit5())
# plot(sol)

# Should make it to the top
θ_end, ω_end = sol.u[end]
x_end, y_end = sin(θ_end), -cos(θ_end)
@test maximum(abs, [x_end, y_end, ω_end] .- [0.0, 1.0, 0.0]) < 1e-3

# Starting at a random point
x0 = lb .+ rand(rng, Float32, 2) .* (ub .- lb)
ode_prob = ODEProblem(closed_loop_dynamics, x0, 75.0f0, p)
sol = solve(ode_prob, Tsit5())
# plot(sol)

# Should make it to the top
θ_end, ω_end = sol.u[end]
x_end, y_end = sin(θ_end), -cos(θ_end)
@test maximum(abs, [x_end, y_end, ω_end] .- [0.0, 1.0, 0.0]) < 1e-3

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
    minimum(V_predict)),
    ", ",
    maximum(V_predict),
    "]"
)
println(
    "V̇ ∋ [",
    minimum(dVdt_predict),
    ", ",
    max(V̇(upright_equilibrium), maximum(dVdt_predict)),
    "]"
)

# Plot results
using Plots

p1 = plot(
    θs / pi,
    ωs,
    V_predict,
    linetype =
    :contourf,
    title = "V",
    xlabel = "θ/π",
    ylabel = "ω",
    c = :bone_1
);
p1 = scatter!([-2 * pi, 0, 2 * pi] / pi, [0, 0, 0],
    label = "Downward Equilibria", color = :green, markershape = :+);
p1 = scatter!(
    [-pi, pi] / pi, [0, 0], label = "Upward Equilibria", color = :red, markershape = :x);
p2 = plot(
    θs / pi,
    ωs,
    dVdt_predict,
    linetype = :contourf,
    title = "dV/dt",
    xlabel = "θ/π",
    ylabel = "ω",
    c = :binary
);
p2 = scatter!([-2 * pi, 0, 2 * pi] / pi, [0, 0, 0],
    label = "Downward Equilibria", color = :green, markershape = :+);
p2 = scatter!([-pi, pi] / pi, [0, 0], label = "Upward Equilibria",
    color = :red, markershape = :x, legend = false);
p3 = plot(
    θs / pi,
    ωs,
    dVdt_predict .< 0,
    linetype = :contourf,
    title = "dV/dt < 0",
    xlabel = "θ/π",
    ylabel = "ω",
    colorbar = false,
    linewidth = 0
);
p3 = scatter!([-2 * pi, 0, 2 * pi] / pi, [0, 0, 0],
    label = "Downward Equilibria", color = :green, markershape = :+);
p3 = scatter!([-pi, pi] / pi, [0, 0], label = "Upward Equilibria",
    color = :red, markershape = :x, legend = false);
plot(p1, p2, p3)
=#
