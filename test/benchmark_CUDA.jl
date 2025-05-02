using NeuralLyapunov
using ModelingToolkit, NeuralPDE
using Lux, LuxCUDA, ComponentArrays
using Boltz.Layers: MLP, PeriodicEmbedding, PositiveDefinite, ShiftTo
using StableRNGs, Random
using OptimizationOptimisers: Adam
using OptimizationOptimJL: BFGS
using Test

rng = StableRNG(0)
Random.seed!(200)

######################### Define dynamics and domain ##########################

@parameters ζ ω_0
defaults = Dict([ζ => 5.0, ω_0 => 1.0])

@independent_variables t
@variables θ(t) τ(t) [input = true]
Dt = Differential(t)
DDt = Dt^2

eqs = [DDt(θ) + 2ζ * ω_0 * Dt(θ) + ω_0^2 * sin(θ) ~ τ]

@named pendulum_driven = ODESystem(
    eqs,
    t,
    [θ, τ],
    [ζ, ω_0];
    defaults = defaults
)

bounds = [
    θ ∈ (0, 2π),
    Dt(θ) ∈ (-2.0, 2.0)
]

upright_equilibrium = [π, 0.0]

####################### Specify neural Lyapunov problem #######################

# Define embedding layer that is periodic with period 2π with respect to θ
periodic_embedding_layer = PeriodicEmbedding([1], [2π])
ps, st = Lux.setup(rng, periodic_embedding_layer)
periodic_embedding = (x) -> first(periodic_embedding_layer(x, ps, st))
fixed_point_embedded = periodic_embedding(upright_equilibrium)

# Define neural Lyapunov structure: Lyapunov-net
dim_state = length(bounds)
dim_hidden = 10

chain = [
    Chain(
        periodic_embedding,
        PositiveDefinite(
            MLP(dim_state + 1, (dim_hidden, dim_hidden, dim_hidden), tanh),
            fixed_point_embedded
        )
    ),
    Chain(
        periodic_embedding,
        ShiftTo(
            MLP(dim_state + 1, (dim_hidden, dim_hidden, 1), tanh),
            fixed_point_embedded,
            [0.0]
        )
    )
]
const gpud = gpu_device()
ps = Lux.initialparameters(rng, chain) .|> ComponentArray .|> gpud |> f32

structure = UnstructuredNeuralLyapunov()
structure = add_policy_search(structure, 1)

# Define neural Lyapunov minimization condition and a periodic decrease condition
minimization_condition = DontCheckNonnegativity(check_fixed_point = false)
periodic_pos_def = function (state, fixed_point)
    return sum(abs2.(periodic_embedding(state) .- periodic_embedding(fixed_point)))
end
decrease_condition = AsymptoticStability(strength = periodic_pos_def)

# Construct neural Lyapunov specification
spec = NeuralLyapunovSpecification(
    structure,
    minimization_condition,
    decrease_condition
)

# Define optimization parameters
opt = [Adam(0.01), BFGS()]
optimization_args = [:maxiters => 300]

strategy = QuasiRandomTraining(4096)

#################################### Run the benchmark #####################################
res = benchmark(
    pendulum_driven,
    bounds,
    spec,
    chain,
    strategy,
    opt;
    fixed_point = upright_equilibrium,
    simulation_time = 300,
    n = 1000,
    optimization_args,
    endpoint_check = (x) -> ≈(periodic_embedding(x), fixed_point_embedded, atol = 1e-3),
    rng,
    init_params = ps
)

cm = res.confusion_matrix

@test cm.tp == 1000
