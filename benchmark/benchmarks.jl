using NeuralLyapunov, BenchmarkTools
using NeuralPDE, Lux, ComponentArrays, ModelingToolkit
import Boltz.Layers: PeriodicEmbedding, MLP
using OptimizationOptimisers: Adam
using StableRNGs, Random
using LinearAlgebra, ForwardDiff

const SUITE = BenchmarkGroup()

rng = StableRNG(0)
Random.seed!(200)

######################### Define dynamics and domain ##########################

@parameters ζ ω_0
initial_conditions = Dict([ζ => 0.5, ω_0 => 1.0])

@independent_variables t
@variables θ(t)
Dt = Differential(t)
DDt = Dt^2

eqs = [DDt(θ) + 2ζ * ω_0 * Dt(θ) + ω_0^2 * sin(θ) ~ 0.0]

@mtkcompile dynamics = System(eqs, t, [θ], [ζ, ω_0]; initial_conditions)

bounds = [θ ∈ (-π, π), Dt(θ) ∈ (-10.0, 10.0)]
lb = [-π, -10.0]
ub = [π, 10.0]
p = [initial_conditions[param] for param in parameters(dynamics)]

f = ODEFunction(dynamics)

dim_state = length(bounds)
dim_hidden = 15
dim_output = 2
chain = [
    Chain(PeriodicEmbedding([1], [2π]), MLP(dim_state + 1, (dim_hidden, dim_hidden, 1), tanh)) for
        _ in 1:dim_output
]
ps, st = Lux.setup(rng, chain)
ps = ps |> ComponentArray |> f64
st = st |> f64

strategy = QuasiRandomTraining(500)
discretization = PhysicsInformedNN(chain, strategy; init_params = ps, init_states = st)

periodic_pos_def = function (state, fixed_point)
    θ, ω = state
    θ_eq, ω_eq = fixed_point
    return (sin(θ) - sin(θ_eq))^2 + (cos(θ) - cos(θ_eq))^2 + (ω - ω_eq)^2
end
structure = PositiveSemiDefiniteStructure(
    dim_output; pos_def = (x, x0) -> log(1.0 + periodic_pos_def(x, x0))
)
minimization_condition = DontCheckNonnegativity(; check_fixed_point = false)
decrease_condition = AsymptoticStability(;
    strength = periodic_pos_def,
    rectifier = (t) -> log(one(t) + exp(t)),
)
spec = NeuralLyapunovSpecification(structure, minimization_condition, decrease_condition)

# =============================================================================
# Problem construction
# =============================================================================

SUITE["construction"] = BenchmarkGroup()

SUITE["construction"]["pde_system"] = @benchmarkable NeuralLyapunovPDESystem(
    $f, $lb, $ub, $spec; p = $p, name = :bench_pde
)

@named pde_system = NeuralLyapunovPDESystem(f, lb, ub, spec; p)

SUITE["construction"]["symbolic_discretize"] = @benchmarkable symbolic_discretize(
    $pde_system, $discretization
)
SUITE["construction"]["discretize"] = @benchmarkable discretize(
    $pde_system, $discretization
) seconds = 300

# =============================================================================
# Short optimization solve — bounded by maxiters, exercises the training path
# =============================================================================

prob = discretize(pde_system, discretization)

SUITE["solve"] = BenchmarkGroup()

SUITE["solve"]["adam_5iters"] = @benchmarkable solve(
    $prob, Adam(0.1); maxiters = 5
) seconds = 240

# =============================================================================
# Lyapunov function extraction + evaluation (untrained initial params)
# =============================================================================

V, V̇ = get_numerical_lyapunov_function(
    discretization.phi, prob.u0.depvar, structure, f,
    zeros(length(bounds)); p
)

states = reduce(hcat, ([x, y] for x in range(-3, 3, 20) for y in range(-5, 5, 20)))

SUITE["evaluation"] = BenchmarkGroup()

SUITE["evaluation"]["V"] = @benchmarkable $V($states)
SUITE["evaluation"]["V̇"] = @benchmarkable $V̇($states)
