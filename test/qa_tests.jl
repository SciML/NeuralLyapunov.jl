using Test, NeuralLyapunov

@testset "Aqua: Quality Assurance" begin
    using Aqua

    Aqua.test_all(NeuralLyapunov)
end
