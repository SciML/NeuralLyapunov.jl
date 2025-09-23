using NeuralPDE, NeuralLyapunov, NeuralLyapunovProblemLibrary
using OptimizationOptimisers: Adam
using OptimizationOptimJL: BFGS
using Random
using Lux, LuxCUDA, ComponentArrays
using Boltz.Layers: ShiftTo, MLP, PeriodicEmbedding
using Test, LinearAlgebra, StableRNGs
using DiffEqGPU: EnsembleGPUArray

const gpud = gpu_device()

###################### Damped simple harmonic oscillator ######################
@testset "Simple harmonic oscillator benchmarking (CUDA training, CPU evaluation)" begin
    println("Benchmark: Damped Simple Harmonic Oscillator (CUDA training, CPU evaluation)")

    rng = StableRNG(0)
    Random.seed!(200)

    # Define dynamics and domain
    function f(state, p, t)
        pos = state[1]
        vel = state[2]
        return vcat(vel, -vel - pos)
    end
    lb = [-2.0, -2.0]
    ub = [2.0, 2.0]
    fixed_point = [0.0, 0.0]
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

    # Define neural Lyapunov structure and minimization condition
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
    opt = [Adam(0.01), Adam(), BFGS()]
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

@testset "Simple harmonic oscillator benchmarking (CPU training, CUDA evaluation)" begin
    println("Benchmark: Damped Simple Harmonic Oscillator (CPU training, CUDA evaluation)")

    Random.seed!(200)

    # Define dynamics and domain
    function sho(x, p, t)
        ζ, ω_0 = p
        pos, vel = x
        return [vel, -2ζ * ω_0 * vel - ω_0^2 * pos]
    end
    function sho(dx, x, p, t)
        ζ, ω_0 = p
        pos, vel = x
        dx[1] = vel
        dx[2] = -2ζ * ω_0 * vel - ω_0^2 * pos
        nothing
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
        ensemble_alg = EnsembleGPUArray(LuxCUDA.CUDABackend())
    )
    cm = out.confusion_matrix

    # SHO is globally asymptotically stable
    @test cm.n == 0

    # Should accurately classify
    @test cm.fn / cm.p < 0.5
end

@testset "Simple harmonic oscillator benchmarking (CUDA training + evaluation)" begin
    println("Benchmark: Damped Simple Harmonic Oscillator (CUDA training + evaluation)")

    rng = StableRNG(0)
    Random.seed!(200)

    # Define dynamics and domain
    function f(x, p, t)
        pos = x[1]
        vel = x[2]
        return vcat(vel, -vel - pos)
    end
    function f(dx, x, p, t)
        pos = x[1]
        vel = x[2]
        dx[1] = vel
        dx[2] = -vel - pos
        nothing
    end
    lb = [-2.0, -2.0]
    ub = [2.0, 2.0]
    fixed_point = [0.0, 0.0]
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

    # Define neural Lyapunov structure and minimization condition
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
    opt = [Adam(0.01), Adam(), BFGS()]
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
        init_states = st,
        ensemble_alg = EnsembleGPUArray(LuxCUDA.CUDABackend())
    )

    # AdditiveLyapunovNet should have no trouble with the globally stable damped SHO, so we
    # expect it to correctly classify everything as within the region of attraction.
    @test out.confusion_matrix.p + out.confusion_matrix.n == out.confusion_matrix.tp
end

####################### Inverted pendulum policy search #######################
@testset "Policy search on inverted pendulum benchmarking (CUDA training)" begin
    println("Benchmark: Inverted Pendulum - Policy Search (CUDA training)")

    rng = StableRNG(0)
    Random.seed!(0)

    # Define dynamics and domain
    p = Float32[0.5f0, 1.0f0]
    @named driven_pendulum = Pendulum(; driven = true, defaults = p)
    t, = independent_variables(driven_pendulum)
    θ, τ = unknowns(driven_pendulum)

    Dt = Differential(t)
    bounds = [
        θ ∈ (0, 2π),
        Dt(θ) ∈ (-2.0f0, 2.0f0)
    ]

    upright_equilibrium = Float32[π, 0.0f0]

    # Define embedding layer that is periodic with period 2π with respect to θ
    # Note: RNG used doesn't matter since the embedding is deterministic
    periodic_embedding_layer = PeriodicEmbedding([1], Float32[2π])
    _ps, _st = Lux.setup(Random.default_rng(), periodic_embedding_layer)
    periodic_embedding(x) = first(periodic_embedding_layer(x, _ps, _st))
    fixed_point_embedded = periodic_embedding(upright_equilibrium)

    # Define neural network discretization
    dim_state = length(bounds)
    dim_hidden = 10
    dim_output = 3
    dim_u = 1
    chain = [
        Chain(
            periodic_embedding_layer,
            AdditiveLyapunovNet(
                MLP(dim_state + 1, (dim_hidden, dim_hidden, dim_output), tanh);
                dim_ϕ = dim_output,
                fixed_point = fixed_point_embedded
            )
        ),
        Chain(
            periodic_embedding_layer,
            ShiftTo(
                MLP(dim_state + 1, (dim_hidden, dim_hidden, dim_u), tanh),
                fixed_point_embedded,
                [0.0f0]
            )
        )
    ]
    ps, st = Lux.setup(rng, chain)
    ps = ps .|> ComponentArray |> gpud |> f32
    st = st |> gpud |> f32

    # Define neural network discretization
    strategy = QuasiRandomTraining(10000)

    # Define neural Lyapunov structure and minimization condition
    structure = NoAdditionalStructure()
    structure = add_policy_search(structure, dim_u)

    minimization_condition = DontCheckNonnegativity(check_fixed_point = false)

    # Define Lyapunov decrease condition
    decrease_condition = AsymptoticStability(
        strength = function (state, fixed_point)
        return sum(abs2, periodic_embedding(state) .- periodic_embedding(fixed_point))
    end
    )

    # Construct neural Lyapunov specification
    spec = NeuralLyapunovSpecification(
        structure,
        minimization_condition,
        decrease_condition
    )

    # Define optimization parameters
    opt = [Adam(0.1f0), BFGS()]
    optimization_args = [[:maxiters => 500], [:maxiters => 500]]

    # Run benchmark
    endpoint_check = (x) -> ≈(periodic_embedding(x), fixed_point_embedded, atol = 0.01f0)
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
