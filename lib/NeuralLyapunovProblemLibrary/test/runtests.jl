using SafeTestsets

@time begin
    @time @safetestset "NeuralLyapunovProblemLibrary.jl" begin
        include("pendulum_test.jl")
    end

    @time @safetestset "NeuralLyapunovProblemLibrary.jl" begin
        include("double_pendulum_test.jl")
    end
end
