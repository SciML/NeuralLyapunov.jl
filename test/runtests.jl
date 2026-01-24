using SafeTestsets

const GROUP = lowercase(get(ENV, "GROUP", "all"))
const DEVICE = lowercase(get(ENV, "DEVICE", "cpu"))

@time begin
    if GROUP == "all" || GROUP == "core"
        if DEVICE == "cpu"
            @time @safetestset "Damped simple harmonic oscillator" begin
                include("damped_sho.jl")
            end
            @time @safetestset "Damped pendulum" begin
                include("damped_pendulum.jl")
            end
            @time @safetestset "Damped pendulum - AdditiveLyapunovNet structure" begin
                include("damped_pendulum_lux.jl")
            end
            @time @safetestset "Damped pendulum - MultiplicativeLyapunovNet structure" begin
                include("damped_pendulum_lux_2.jl")
            end
        end

        if DEVICE == "gpu"
            @time @safetestset "CUDA test - Damped SHO" begin
                include("damped_sho_CUDA.jl")
            end
        end
    end

    if GROUP == "all" || GROUP == "policy_search"
        if DEVICE == "cpu"
            @time @safetestset "Policy search - inverted pendulum" begin
                include("inverted_pendulum.jl")
            end
            @time @safetestset "Policy search - inverted pendulum (ODESystem)" begin
                include("inverted_pendulum_ODESystem.jl")
            end
        end
    end

    if GROUP == "all" || GROUP == "roa"
        if DEVICE == "cpu"
            @time @safetestset "Region of attraction estimation" begin
                include("roa_estimation.jl")
            end
        end
    end

    if GROUP == "all" || GROUP == "local_lyapunov"
        if DEVICE == "cpu"
            @time @safetestset "Local Lyapunov function search" begin
                include("local_lyapunov.jl")
            end
        end
    end

    if GROUP == "all" || GROUP == "benchmarking"
        if DEVICE == "cpu"
            @time @safetestset "Benchmarking tool" begin
                include("benchmark.jl")
            end
        end
        if DEVICE == "gpu"
            @time @safetestset "Benchmarking tool - CUDA" begin
                include("benchmark_CUDA.jl")
            end
        end
    end

    if GROUP == "all" || GROUP == "doctests"
        if DEVICE == "cpu"
            @time @safetestset "Doctests" begin
                include("doctests.jl")
            end
        end
    end

    if GROUP == "all" || GROUP == "unimplemented"
        if DEVICE == "cpu"
            @time @safetestset "Errors for partially-implemented extensions" begin
                include("unimplemented.jl")
            end
        end
    end

    if GROUP == "all" || GROUP == "qa"
        if DEVICE == "cpu"
            @time @safetestset "Quality Assurance" begin
                include("qa_tests.jl")
            end
        end
    end

    if GROUP == "all" || GROUP == "jet"
        if DEVICE == "cpu"
            @time @safetestset "JET Static Analysis" begin
                include("jet_tests.jl")
            end
        end
    end
end
