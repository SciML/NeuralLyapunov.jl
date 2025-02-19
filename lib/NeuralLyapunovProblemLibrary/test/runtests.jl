using SafeTestsets

@time @safetestset "NeuralLyapunovProblemLibrary.jl" begin
    include("pendulum_test.jl")
end
