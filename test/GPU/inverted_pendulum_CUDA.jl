using NeuralPDE, ModelingToolkit, NeuralLyapunov, NeuralLyapunovProblemLibrary
using OrdinaryDiffEqTsit5: Tsit5
using ModelingToolkit: unbound_inputs
using SciMLBase: ODEFunction, ODEInputFunction, ODEProblem, solve, symbolic_discretize
using Lux, LuxCUDA, ComponentArrays
using Boltz.Layers: ShiftTo, MLP, PeriodicEmbedding
import Optimization
using OptimizationOptimisers: Adam
using StableRNGs, Random
using Test, LinearAlgebra, ForwardDiff

rng = StableRNG(0)
Random.seed!(200)

println("Inverted Pendulum - Policy Search (CUDA)")

######################### Define dynamics and domain ##########################

ζ = 0.5f0
ω0 = 1.0f0
p = Float32[ζ, ω0]
@named driven_pendulum = Pendulum(; driven = true, defaults = p)
τ, = unbound_inputs(driven_pendulum)
driven_pendulum = mtkcompile(driven_pendulum; inputs = [τ], split = false)
θ, ω = unknowns(driven_pendulum)

bounds = [
    θ ∈ Float32.((0, 2π)),
    ω ∈ (-5ω0, 5ω0),
]

upright_equilibrium = Float32[π, 0.0f0]

####################### Specify neural Lyapunov problem #######################

# Define embedding layer that is periodic with period 2π with respect to θ
# Note: RNG used doesn't matter since the embedding is deterministic
periodic_embedding_layer = PeriodicEmbedding([1], Float32[2π])
_ps, _st = Lux.setup(Random.default_rng(), periodic_embedding_layer)
periodic_embedding(x) = first(periodic_embedding_layer(x, _ps, _st))
fixed_point_embedded = periodic_embedding(upright_equilibrium)

# Define neural network discretization
dim_state = length(bounds)
dim_hidden = 20
dim_phi = 10
dim_u = 1
u0 = Float32[0.0f0]
chain = [
    Chain(
        periodic_embedding_layer,
        AdditiveLyapunovNet(
            MLP(dim_state + 1, (dim_hidden, dim_hidden, dim_hidden, dim_phi), tanh);
            dim_ϕ = dim_phi,
            fixed_point = fixed_point_embedded
        )
    ),
    Chain(
        periodic_embedding_layer,
        ShiftTo(
            MLP(dim_state + 1, (dim_hidden, dim_hidden, dim_hidden, dim_u), tanh),
            fixed_point_embedded,
            u0
        )
    ),
]

const gpud = gpu_device()
ps, st = Lux.setup(rng, chain)
ps = ps .|> ComponentArray |> gpud |> f32
st = st |> gpud |> f32

# Define neural network discretization
strategy = QuasiRandomTraining(10000)
discretization = PhysicsInformedNN(chain, strategy; init_params = ps, init_states = st)

# Define neural Lyapunov structure and corresponding minimization condition
structure = add_policy_search(NoAdditionalStructure(), dim_u)

minimization_condition = DontCheckNonnegativity(check_fixed_point = false)

# Define a periodic Lyapunov decrease condition
decrease_condition = AsymptoticStability(
    strength = function (state, fixed_point)
        return sum(abs2, periodic_embedding(state) .- periodic_embedding(fixed_point))
    end
)

# Construct neural Lyapunov specification
spec = NeuralLyapunovSpecification(structure, minimization_condition, decrease_condition)

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

res = Optimization.solve(prob, Adam(1.0f-2); maxiters = 500)
prob = Optimization.remake(prob; u0 = res.u)
res = Optimization.solve(prob, Adam(1.0f-4); maxiters = 500)

########################### Get numerical functions ###########################

net = discretization.phi
_θ = res.u.depvar

open_loop_pendulum_dynamics = ODEInputFunction(driven_pendulum)

(V, V̇) = get_numerical_lyapunov_function(
    net,
    _θ,
    structure,
    open_loop_pendulum_dynamics,
    upright_equilibrium;
    p
)

u = get_policy(net, _θ, 1, dim_u; fixed_point = upright_equilibrium)

const cpud = cpu_device()
closed_loop_dynamics = ODEFunction(
    (x, p, t) -> open_loop_pendulum_dynamics(x, u(x) |> cpud, p, t);
    sys = driven_pendulum
)

################################## Simulate ###################################

lb = Float32[0.0, -2ω0];
ub = Float32[2π, 2ω0];
θs = (-2.0f0 * π):0.02f0:(2.0f0 * π)
ωs = lb[2]:0.02f0:ub[2]
states = mapreduce(collect, hcat, Iterators.product(θs, ωs))
V_samples = vec(V(states))
V̇_samples = vec(V̇(states))

#################################### Tests ####################################

# Network structure should enforce positive definiteness
V0 = only(V(upright_equilibrium) |> cpud)
@test V0 == 0.0
@test min(V0, minimum(V_samples)) ≥ 0.0
@test maximum(abs, ForwardDiff.jacobian(V, upright_equilibrium)) < 2.0e-5
@test minimum(eigvals(ForwardDiff.hessian((x) -> only(V(x) |> cpud), upright_equilibrium))) ≥ 0

# Network structure should enforce periodicity in θ
x0 = (ub .- lb) .* rand(rng, Float32, 2, 100) .+ lb
@test maximum(abs, V(x0 .+ Float32[2π, 0.0]) .- V(x0)) < 1.0e-3

# Training should result in a locally stable fixed point at the upright equilibrium
# Check for approximately zero angular acceleration
ẋ0 = closed_loop_dynamics(upright_equilibrium, p, 0.0)
@test abs(ẋ0[2]) < 3.0e-6
# Check for nonpositive eigenvalues of the Jacobian
@test_broken maximum(
    eigvals(
        ForwardDiff.jacobian((x) -> closed_loop_dynamics(x, p, 0.0), upright_equilibrium)
    )
) ≤ 0

# Check for local negative definiteness of V̇
V̇0 = only(V̇(upright_equilibrium) |> cpud)
@test abs(V̇0) < 1.0e-12
@test maximum(abs, ForwardDiff.jacobian(V̇, upright_equilibrium)) < 2.0e-5
@test maximum(
    eigvals(ForwardDiff.hessian((x) -> only(V̇(x) |> cpud), upright_equilibrium))
) ≤ 0

# V̇ should be negative almost everywhere
@test sum(V̇_samples .> 0) / length(V_samples) < 0.01

################################## Simulate ###################################

# Starting still at bottom ...
downward_equilibrium = zeros(Float32, 2)
ode_prob = ODEProblem(closed_loop_dynamics, downward_equilibrium, 120.0f0, p)
sol = solve(ode_prob, Tsit5())
# plot(sol)

# ...the system should make it to the top
θ_end, ω_end = sol.u[end]
x_end, y_end = sin(θ_end), -cos(θ_end)
@test_broken maximum(abs, [x_end, y_end, ω_end] .- [0.0, 1.0, 0.0]) < 2.0e-2

# Starting at a random point ...
x0 = lb .+ rand(rng, Float32, 2) .* (ub .- lb)
ode_prob = ODEProblem(closed_loop_dynamics, x0, 150.0f0, p)
sol = solve(ode_prob, Tsit5())
# plot(sol)

# ...the system should make it to the top
θ_end, ω_end = sol.u[end]
x_end, y_end = sin(θ_end), -cos(θ_end)
@test_broken maximum(abs, [x_end, y_end, ω_end] .- [0.0, 1.0, 0.0]) < 2.0e-2

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
    θs / pi,
    ωs,
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
    θs / pi,
    ωs,
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
    θs / pi,
    ωs,
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
