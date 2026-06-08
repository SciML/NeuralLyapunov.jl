using Pkg
using SafeTestsets: @safetestset

const GROUP = get(ENV, "NEURALLYAPUNOV_TEST_GROUP", get(ENV, "GROUP", "All"))

function activate_qa_env()
    Pkg.activate(joinpath(@__DIR__, "qa"))
    return Pkg.instantiate()
end

@time begin
    if GROUP == "All" || GROUP == "Core" || GROUP == "Pendula"
        @time @safetestset "Simple pendulum" begin
            include("pendulum_test.jl")
        end

        @time @safetestset "Double pendulum" begin
            include("double_pendulum_test.jl")
        end
    end

    if GROUP == "All" || GROUP == "Core" || GROUP == "Quadrotors"
        @time @safetestset "Planar Quadrotor" begin
            include("planar_quadrotor_test.jl")
        end

        @time @safetestset "Quadrotor (3D)" begin
            include("quadrotor_test.jl")
        end
    end

    # The following test run in different GitHub actions, so aren't in the "Core" group
    if GROUP == "All" || GROUP == "QA"
        activate_qa_env()
        @time @safetestset "Quality Assurance" begin
            include("qa/qa.jl")
        end
        @time @safetestset "Explicit Imports" begin
            include("qa/explicit_imports.jl")
        end
    end

    if GROUP == "All" || GROUP == "Doctests"
        @time @safetestset "Doctests" begin
            include("doctests.jl")
        end

    end
end
