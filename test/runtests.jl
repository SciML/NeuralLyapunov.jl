using SafeTestsets: @safetestset
using SciMLTesting

run_tests(;
    core = function ()
        @time @safetestset "Damped simple harmonic oscillator" begin
            include(joinpath(@__DIR__, "Core", "damped_sho.jl"))
        end
        @time @safetestset "Damped pendulum" begin
            include(joinpath(@__DIR__, "Core", "damped_pendulum.jl"))
        end
        @time @safetestset "Damped pendulum - AdditiveLyapunovNet structure" begin
            include(joinpath(@__DIR__, "Core", "damped_pendulum_lux.jl"))
        end
        return @time @safetestset "Damped pendulum - MultiplicativeLyapunovNet structure" begin
            include(joinpath(@__DIR__, "Core", "damped_pendulum_lux_2.jl"))
        end
    end,
    groups = Dict(
        "Policy_search" => function ()
            @time @safetestset "Policy search - inverted pendulum" begin
                include(joinpath(@__DIR__, "Policy_search", "inverted_pendulum.jl"))
            end
            return @time @safetestset "Policy search - inverted pendulum (ODESystem)" begin
                include(joinpath(@__DIR__, "Policy_search", "inverted_pendulum_ODESystem.jl"))
            end
        end,
        "ROA" => function ()
            return @time @safetestset "Region of attraction estimation" begin
                include(joinpath(@__DIR__, "ROA", "roa_estimation.jl"))
            end
        end,
        "Local_lyapunov" => function ()
            return @time @safetestset "Local Lyapunov function search" begin
                include(joinpath(@__DIR__, "Local_lyapunov", "local_lyapunov.jl"))
            end
        end,
        "Benchmarking" => function ()
            return @time @safetestset "Benchmarking tool" begin
                include(joinpath(@__DIR__, "Benchmarking", "benchmark.jl"))
            end
        end,
        "Unimplemented" => function ()
            return @time @safetestset "Errors for partially-implemented extensions" begin
                include(joinpath(@__DIR__, "Unimplemented", "unimplemented.jl"))
            end
        end,
        "Doctests" => (;
            env = joinpath(@__DIR__, "Doctests"),
            body = function ()
                return @time @safetestset "Doctests" begin
                    include(joinpath(@__DIR__, "Doctests", "doctests.jl"))
                end
            end,
        ),
        "GPU" => (;
            env = joinpath(@__DIR__, "GPU"),
            body = function ()
                @time @safetestset "CUDA test - Damped SHO" begin
                    include(joinpath(@__DIR__, "GPU", "damped_sho_CUDA.jl"))
                end
                return @time @safetestset "Benchmarking tool - CUDA" begin
                    include(joinpath(@__DIR__, "GPU", "benchmark_CUDA.jl"))
                end
            end,
        ),
    ),
    qa = (;
        env = joinpath(@__DIR__, "QA"),
        body = function ()
            @time @safetestset "Quality Assurance" begin
                include(joinpath(@__DIR__, "QA", "qa.jl"))
            end
            return @time @safetestset "Explicit Imports" begin
                include(joinpath(@__DIR__, "QA", "explicit_imports.jl"))
            end
        end,
    ),
    all = ["Core", "Policy_search", "ROA", "Local_lyapunov", "Benchmarking", "Unimplemented"],
    sublib_env = "NEURALLYAPUNOV_TEST_GROUP",
    lib_dir = joinpath(dirname(@__DIR__), "lib"),
)
