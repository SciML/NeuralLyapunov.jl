using NeuralPDE, NeuralLyapunov, SciMLBase
using OptimizationOptimisers: Adam
using Random
using Lux, LuxCUDA, ComponentArrays
using MLDataDevices: CUDADevice, get_device_type
using Boltz.Layers: MLP
using Test, StableRNGs
using DiffEqGPU: EnsembleGPUArray

const gpud = CUDADevice()

###################### Damped simple harmonic oscillator ######################
@testset "GPU-resident init_params are rejected" begin
    println("Benchmark: GPU-resident init_params")

    rng = StableRNG(0)
    Random.seed!(200)

    # Define dynamics and domain
    function f(state, p, t)
        pos = state[1]
        vel = state[2]
        return vcat(vel, -vel - pos)
    end
    lb = Float32[-2.0, -2.0]
    ub = Float32[2.0, 2.0]
    fixed_point = Float32[0.0, 0.0]
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
    @test get_device_type(ps) <: CUDADevice

    # Define training strategy
    strategy = QuasiRandomTraining(2500)

    # Define neural Lyapunov structure and minimization condition
    structure = NoAdditionalStructure()
    minimization_condition = DontCheckNonnegativity()

    # Define Lyapunov decrease condition
    # This damped SHO has exponential decrease at a rate of k = 0.5, so we train to certify that
    decrease_condition = ExponentialStability(0.5f0)

    # Construct neural Lyapunov specification
    spec = NeuralLyapunovSpecification(
        structure,
        minimization_condition,
        decrease_condition
    )

    # Define optimization parameters
    opt = [Adam(1.0f-2), Adam(1.0f-3), Adam(1.0f-5)]
    optimization_args = [:maxiters => 300]

    # NeuralPDE 7 cannot train a PDESystem on the GPU (SciML/NeuralPDE.jl#1161)
    @test_throws ArgumentError benchmark(
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
        init_params = ps
    )
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
    structure = NonnegativeStructure(dim_output; δ = 1.0e-6)
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
    opt = Adam(1.0e-3)
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
    @test sum(cm.Count[2:3]) == 0

    # Should accurately classify
    @test cm.Count[4] / sum(cm.Count) < 0.5
end
