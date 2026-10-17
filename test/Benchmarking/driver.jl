using BenchmarkTools, Test

@testset "Benchmark suite entries" begin
    include(joinpath(@__DIR__, "..", "..", "benchmark", "benchmarks.jl"))

    @test Set(keys(SUITE)) == Set(["construction", "solve", "evaluation"])
    for group in values(SUITE), benchmark in values(group)
        @test run(benchmark; samples = 1, evals = 1, seconds = 0.001) isa BenchmarkTools.Trial
    end
end
