using SafeTestsets

const GROUP = lowercase(get(ENV, "GROUP", "all"))

@time begin
    if GROUP == "all" || GROUP == "core"
        @time @safetestset "Damped simple harmonic oscillator" begin
            include("damped_sho.jl")
        end
        @time @safetestset "Damped pendulum" begin
            include("damped_pendulum.jl")
        end
    end

    if GROUP == "all" || GROUP == "policy_search"
        @time @safetestset "Policy search - inverted pendulum" begin
            include("inverted_pendulum.jl")
        end
        @time @safetestset "Policy search - inverted pendulum (ODESystem)" begin
            include("inverted_pendulum_ODESystem.jl")
        end
    end

    if GROUP == "all" || GROUP == "roa"
        @time @safetestset "Region of attraction estimation" begin
            include("roa_estimation.jl")
        end
    end

    if GROUP == "all" || GROUP == "local_lyapunov"
        @time @safetestset "Local Lyapunov function search" begin
            include("local_lyapunov.jl")
        end
    end

    if GROUP == "all" || GROUP == "cuda"
        @time @safetestset "CUDA test - Damped SHO" begin
            include("damped_sho_CUDA.jl")
        end
    end

    if GROUP == "all" || GROUP == "unimplemented"
        @time @safetestset "Errors for partially-implemented extensions" begin
            include("unimplemented.jl")
        end
    end

    if GROUP == "all" || GROUP == "benchmarking"
        @time @safetestset "Benchmarking tool" begin
            include("benchmark.jl")
        end
    end
end
