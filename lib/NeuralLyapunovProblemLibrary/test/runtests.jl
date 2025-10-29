using Test, SafeTestsets

const GROUP = lowercase(get(ENV, "GROUP", "all"))

@time @testset "NeuralLyapunovProblemLibrary" begin
    if GROUP == "all" || GROUP == "pendula"
        @time @safetestset "Simple pendulum" begin
            include("pendulum_test.jl")
        end

        @time @safetestset "Double pendulum" begin
            include("double_pendulum_test.jl")
        end
    end

    if GROUP == "all" || GROUP == "quadrotors"
        @time @safetestset "Planar Quadrotor" begin
            include("planar_quadrotor_test.jl")
        end

        @time @safetestset "Quadrotor (3D)" begin
            include("quadrotor_test.jl")
        end
    end

    if GROUP == "all" || GROUP == "qa"
        @time @safetestset "Quality Assurance" begin
            include("qa_tests.jl")
        end
    end
end
