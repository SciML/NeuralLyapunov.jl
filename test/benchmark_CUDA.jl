using NeuralPDE, NeuralLyapunov, NeuralLyapunovProblemLibrary
import Optimization, OptimizationOptimisers, OptimizationOptimJL
using Random
using Lux, LuxCUDA, ComponentArrays
using Boltz.Layers: ShiftTo, MLP, PeriodicEmbedding
using Test, LinearAlgebra, StableRNGs

const gpud = gpu_device()

###################### Damped simple harmonic oscillator ######################
@testset "Simple harmonic oscillator benchmarking (CUDA)" begin
    println("Benchmark: Damped Simple Harmonic Oscillator (CUDA)")

    rng = StableRNG(0)
    Random.seed!(200)

    # Define dynamics and domain

    "Simple Harmonic Oscillator Dynamics"
    function f(state, p, t)
        pos = state[1]
        vel = state[2]
        vcat(vel, -vel - pos)
    end
    lb = [-2.0, -2.0];
    ub = [2.0, 2.0];
    fixed_point = [0.0, 0.0];
    dynamics = ODEFunction(f; sys = SciMLBase.SymbolCache([:x, :v]))

    # Specify neural Lyapunov problem

    # Define neural network discretization
    dim_state = length(lb)
    dim_hidden = 20
    chain = AdditiveLyapunovNet(
        MLP(dim_state, (dim_hidden, dim_hidden, dim_hidden, 1), tanh);
        dim_ϕ = 1,
        fixed_point
    )
    ps, st = Lux.setup(rng, chain)
    ps = ps |> ComponentArray |> gpud |> f32
    st = st |> gpud |> f32

    # Define training strategy
    strategy = QuasiRandomTraining(2500)
    discretization = PhysicsInformedNN(chain, strategy; init_params = ps, init_states = st)

    # Define neural Lyapunov structure
    structure = NoAdditionalStructure()
    minimization_condition = DontCheckNonnegativity()

    # Define Lyapunov decrease condition
    # This damped SHO has exponential decrease at a rate of k = 0.5, so we train to certify that
    decrease_condition = ExponentialStability(0.5)

    # Construct neural Lyapunov specification
    spec = NeuralLyapunovSpecification(
        structure,
        minimization_condition,
        decrease_condition
    )

    # Benchmarking
    # Define optimization parameters
    opt = [
        OptimizationOptimisers.Adam(0.01),
        OptimizationOptimisers.Adam(),
        OptimizationOptimJL.BFGS()
    ]
    optimization_args = [:maxiters => 300]

    out = benchmark(
        dynamics,
        lb,
        ub,
        spec,
        chain,
        strategy,
        opt;
        fixed_point,
        simulation_time = 300,
        n = 1000,
        optimization_args,
        rng,
        init_params = ps,
        init_states = st
    )

    # AdditiveLyapunovNet should have no trouble with the globally stable damped SHO, so we
    # expect it to correctly classify everything as within the region of attraction.
    @test out.confusion_matrix.p + out.confusion_matrix.n == out.confusion_matrix.tp
end

####################### Inverted pendulum policy search #######################
@testset "Policy search on inverted pendulum benchmarking (CUDA)" begin
    println("Benchmark: Inverted Pendulum - Policy Search (CUDA)")

    rng = StableRNG(0)
    Random.seed!(200)

    # Define dynamics and domain
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

    # Define embedding layer that is periodic with period 2π with respect to θ
    # Note: RNG used doesn't matter since the embedding is deterministic
    periodic_embedding_layer = PeriodicEmbedding([1], [2π])
    _ps, _st = Lux.setup(Random.default_rng(), periodic_embedding_layer)
    periodic_embedding(x) = first(periodic_embedding_layer(x, _ps, _st))
    fixed_point_embedded = periodic_embedding(upright_equilibrium)

    # Define neural network discretization
    dim_state = length(bounds)
    dim_hidden = 25
    dim_u = 1
    chain = [
        Chain(
            periodic_embedding_layer,
            AdditiveLyapunovNet(
                MLP(dim_state + 1, (dim_hidden, dim_hidden, dim_hidden), tanh);
                dim_ϕ = dim_hidden,
                fixed_point = fixed_point_embedded,
            )
        ),
        Chain(
            periodic_embedding_layer,
            ShiftTo(
                MLP(dim_state + 1, (dim_hidden, dim_hidden, 1), tanh),
                fixed_point_embedded,
                [0.0]
            )
        )
    ]
    ps, st = Lux.setup(rng, chain)
    ps = ps .|> ComponentArray |> gpud |> f32
    st = st |> gpud |> f32

    # Define neural network discretization
    strategy = QuasiRandomTraining(10000)

    # Define neural Lyapunov structure
    periodic_pos_def = function (state, fixed_point)
        θ, ω = state
        θ_eq, ω_eq = fixed_point
        return (sin(θ) - sin(θ_eq))^2 + (cos(θ) - cos(θ_eq))^2 + (ω - ω_eq)^2
    end

    structure = NoAdditionalStructure()
    structure = add_policy_search(structure, dim_u)

    minimization_condition = DontCheckNonnegativity(check_fixed_point = false)

    # Define Lyapunov decrease condition
    decrease_condition = AsymptoticStability(strength = periodic_pos_def)

    # Construct neural Lyapunov specification
    spec = NeuralLyapunovSpecification(
        structure,
        minimization_condition,
        decrease_condition
    )

    # Define optimization parameters
    opt = [OptimizationOptimisers.Adam(0.05), OptimizationOptimJL.BFGS()]
    optimization_args = [[:maxiters => 500], [:maxiters => 500]]

    # Run benchmark
    endpoint_check = (x) -> ≈(periodic_embedding(x), fixed_point_embedded, atol = 0.01)
    out = benchmark(
        driven_pendulum,
        bounds,
        spec,
        chain,
        strategy,
        opt;
        simulation_time = 200,
        n = 200,
        fixed_point = upright_equilibrium,
        optimization_args,
        endpoint_check,
        classifier = (V, V̇, x) -> V̇ < zero(V̇) || endpoint_check(x),
        init_params = ps,
        init_states = st
    )
    cm = out.confusion_matrix

    # Resulting controller should drive more states to equilibrium than not
    @test cm.p > cm.n

    # Resulting classifier should be accurate
    @test (cm.tp + cm.tn) / (cm.p + cm.n) > 0.9
end
