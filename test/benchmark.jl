using NeuralPDE, NeuralLyapunov, Lux, NeuralLyapunovProblemLibrary
import Boltz.Layers: PeriodicEmbedding, MLP
using OptimizationOptimisers: Adam
using OptimizationOptimJL: BFGS
using OrdinaryDiffEq: EnsembleSerial
using StableRNGs, Random
using Test

###################### Damped simple harmonic oscillator ######################
@testset "Simple harmonic oscillator benchmarking" begin
    println("Benchmark: Damped SHO")

    Random.seed!(200)

    # Define dynamics and domain

    "Simple Harmonic Oscillator Dynamics"
    function sho(state, p, t)
        ζ, ω_0 = p
        pos = state[1]
        vel = state[2]
        return vcat(vel, -2ζ * ω_0 * vel - ω_0^2 * pos)
    end
    lb = [-5.0, -2.0]
    ub = [5.0, 2.0]
    p = [0.5, 1.0]
    fixed_point = [0.0, 0.0]
    sho_dynamics = ODEFunction(sho; sys = SciMLBase.SymbolCache([:x, :v], [:ζ, :ω_0]))

    # Define neural network discretization
    dim_state = length(lb)
    dim_hidden = 10
    dim_output = 3
    chain = [MLP(dim_state, (dim_hidden, dim_hidden, 1), tanh) for _ in 1:dim_output]

    # Define training strategy
    strategy = QuasiRandomTraining(1000)

    # Define neural Lyapunov structure and corresponding minimization condition
    structure = NonnegativeStructure(dim_output; δ = 1e-6)
    minimization_condition = DontCheckNonnegativity(check_fixed_point = true)

    # Define Lyapunov decrease condition
    # Damped SHO has exponential decrease at a rate of k = ζ * ω_0, so we train to certify that
    decrease_condition = ExponentialStability(prod(p))

    # Construct neural Lyapunov specification
    spec = NeuralLyapunovSpecification(
        structure,
        minimization_condition,
        decrease_condition
    )

    # Define optimization parameters
    opt = Adam()
    optimization_args = [:maxiters => 450]

    # Run benchmark
    out = benchmark(
        sho_dynamics,
        lb,
        ub,
        spec,
        chain,
        strategy,
        opt;
        simulation_time = 100,
        n = 200,
        p,
        optimization_args,
        ensemble_alg = EnsembleSerial()
    )
    cm = out.confusion_matrix

    # SHO is globally asymptotically stable
    @test sum(cm.Count[2:3]) == 0

    # Should accurately classify
    @test cm.Count[4] / sum(cm.Count[1:2]) < 0.5
end

####################### Inverted pendulum policy search #######################
@testset "Policy search on inverted pendulum benchmarking" begin
    println("Benchmark: Inverted Pendulum - Policy Search")

    Random.seed!(200)

    # Define dynamics and domain
    function open_loop_pendulum_dynamics(x, u, p, t)
        θ, ω = x
        ζ, ω_0 = p
        τ = u[]
        return [ω
                -2ζ * ω_0 * ω - ω_0^2 * sin(θ) + τ]
    end

    lb = [0.0, -2.0]
    ub = [2π, 2.0]
    upright_equilibrium = [π, 0.0]
    p = Float32[0.5, 1.0]
    state_syms = [:θ, :ω]
    parameter_syms = [:ζ, :ω_0]

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
    ps, st = Lux.setup(StableRNG(0), chain)

    # Define neural network discretization
    strategy = QuasiRandomTraining(10000)

    # Define neural Lyapunov structure and corresponding minimization condition
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

    # Define Lyapunov decrease condition
    decrease_condition = AsymptoticStability(strength = periodic_pos_def)

    # Construct neural Lyapunov specification
    spec = NeuralLyapunovSpecification(
        structure,
        minimization_condition,
        decrease_condition
    )

    # Define optimization parameters
    opt = [Adam(0.05), Adam(0.001)]
    optimization_args = [[:maxiters => 300], [:maxiters => 300]]

    # Run benchmark
    endpoint_check = (x) -> ≈([sin(x[1]), cos(x[1]), x[2]], [0, -1, 0], atol = 5e-3)
    classifier = (V, V̇, x) -> V̇ < zero(V̇) || endpoint_check(x)
    benchmarking_results = benchmark(
        open_loop_pendulum_dynamics,
        lb,
        ub,
        spec,
        chain,
        strategy,
        opt;
        simulation_time = 200,
        n = 200,
        fixed_point = upright_equilibrium,
        p,
        optimization_args,
        state_syms,
        parameter_syms,
        policy_search = true,
        endpoint_check,
        classifier,
        init_params = ps,
        init_states = st
    )

    # Resulting controller should drive more states to equilibrium than not
    cm = benchmarking_results.confusion_matrix
    @test cm.Count[1] + cm.Count[4] > sum(cm.Count[2:3])

    # Resulting classifier should be accurate
    @test (cm.Count[1] + cm.Count[3]) / sum(cm.Count) > 0.9

    # Generate numerical Lyapunov function for testing
    (V, V̇) = get_numerical_lyapunov_function(
        benchmarking_results.phi,
        benchmarking_results.θ,
        structure,
        open_loop_pendulum_dynamics,
        upright_equilibrium;
        p
    )

    # Check V samples
    states = benchmarking_results.data[!, "Initial State"]
    V_samples = benchmarking_results.data[!, "V"]
    @test all(V.(states) .== benchmarking_results.V.(states) .== V_samples)

    # Check V̇ samples
    V̇_samples = benchmarking_results.data[!, "dVdt"]
    @test all(V̇.(states) .== benchmarking_results.V̇.(states) .== V̇_samples)

    # Check actual classification
    endpoints = benchmarking_results.data[!, "Final State"]
    actual = benchmarking_results.data[!, "Actually in RoA"]
    @test all(endpoint_check.(endpoints) .== actual)

    # Check predicted classification
    predicted = benchmarking_results.data[!, "Predicted in RoA"]
    @test all(classifier.(V_samples, V̇_samples, states) .== predicted)
end

############################### Damped pendulum ###############################
@testset "Damped pendulum (ODESystem) benchmarking" begin
    println("Benchmark: Damped Pendulum (ODESystem)")

    # Define dynamics and domain
    @named damped_pendulum = Pendulum(; driven = false, defaults = Float32[5.0, 1.0])
    t, = independent_variables(damped_pendulum)
    θ, = unknowns(damped_pendulum)
    Dt = Differential(t)

    damped_pendulum = structural_simplify(damped_pendulum)
    bounds = [
        θ ∈ Float32.((-π, π)),
        Dt(θ) ∈ (-10.0f0, 10.0f0)
    ]

    # Define neural network discretization
    # We use an input layer that is periodic with period 2π with respect to θ
    dim_state = length(bounds)
    dim_hidden = 15
    dim_output = 2
    chain = [Chain(
                 PeriodicEmbedding([1], Float32[2π]),
                 MLP(dim_state + 1, (dim_hidden, dim_hidden, 1), tanh)
             ) for _ in 1:dim_output]

    # Define neural network discretization
    strategy = QuadratureTraining()

    # Define neural Lyapunov structure and corresponding minimization condition
    periodic_pos_def = function (state, fixed_point)
        θ, ω = state
        θ_eq, ω_eq = fixed_point
        return (sin(θ) - sin(θ_eq))^2 + (cos(θ) - cos(θ_eq))^2 + (ω - ω_eq)^2
    end
    structure = PositiveSemiDefiniteStructure(
        dim_output;
        pos_def = (x, x0) -> log(1 + periodic_pos_def(x, x0))
    )
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
    opt = Adam()
    optimization_args = [:maxiters => 600]

    # Run benchmark
    out = benchmark(
        damped_pendulum,
        bounds,
        spec,
        chain,
        strategy,
        opt;
        simulation_time = 300,
        n = 200,
        optimization_args,
        endpoint_check = (x) -> ≈([sin(x[1]), cos(x[1]), x[2]], [0, 1, 0], atol = 1e-3),
        rng = StableRNG(0)
    )
    cm = out.confusion_matrix

    # Damped pendulum is globally asymptotically stable, except at upright equilibrium
    @test sum(cm.Count[2:3]) == 0

    # Should accurately classify
    @test cm.Count[4] / sum(cm.Count[1:2]) < 0.5
end

####################### Inverted pendulum policy search #######################
@testset "Policy search on inverted pendulum (ODESystem) benchmarking" begin
    println("Benchmark: Inverted Pendulum - Policy Search (ODESystem)")

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

    # Define neural network discretization
    # We use an input layer that is periodic with period 2π with respect to θ
    dim_state = length(bounds)
    dim_hidden = 25
    dim_phi = 3
    dim_u = 1
    dim_output = dim_phi + dim_u
    chain = [Chain(
                 PeriodicEmbedding([1], [2π]),
                 MLP(dim_state + 1, (dim_hidden, dim_hidden, 1), tanh)
             ) for _ in 1:dim_output]
    ps, st = Lux.setup(StableRNG(0), chain)
    ps = ps |> f64
    st = st |> f64

    # Define neural network discretization
    strategy = QuasiRandomTraining(10000)

    # Define neural Lyapunov structure and corresponding minimization condition
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

    # Define Lyapunov decrease condition
    decrease_condition = AsymptoticStability(strength = periodic_pos_def)

    # Construct neural Lyapunov specification
    spec = NeuralLyapunovSpecification(
        structure,
        minimization_condition,
        decrease_condition
    )

    # Define optimization parameters
    opt = [Adam(0.05), BFGS()]
    optimization_args = [[:maxiters => 300], [:maxiters => 300]]

    # Run benchmark
    endpoint_check = (x) -> ≈([sin(x[1]), cos(x[1]), x[2]], [0, -1, 0], atol = 5e-3)
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
    @test cm.Count[1] + cm.Count[4] > sum(cm.Count[2:3])

    # Resulting classifier should be accurate
    @test (cm.Count[1] + cm.Count[3]) / sum(cm.Count) > 0.9
end
