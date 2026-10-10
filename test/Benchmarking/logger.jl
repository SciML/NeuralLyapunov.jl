using NeuralLyapunov, Optimization, Test
using OptimizationOptimisers: Adam
using NeuralLyapunov: NeuralLyapunovBenchmarkLogger, LossLoggingCallback, start_pass!

loss(u, p = nothing) = sum(abs2, u)
grad!(g, u, p = nothing) = (g .= 2 .* u)
prob = OptimizationProblem(OptimizationFunction(loss; grad = grad!), [1.0])

# Two Adam passes with the default `save_best = true`, which repeats the callback for the
# final iteration of each pass
function train(log_frequency)
    logger = NeuralLyapunovBenchmarkLogger()
    log_loss = LossLoggingCallback(logger, log_frequency)
    seen = Tuple{Int, Float64}[]
    p = prob
    for _ in 1:2
        start_pass!(log_loss)
        callback = (state, l) -> (push!(seen, (state.iter, l)); log_loss(state, l))
        sol = solve(p, Adam(0.1); maxiters = 2, callback)
        p = remake(p; u0 = sol.u)
    end
    return logger, seen
end

@testset "Loss logging follows solver iterations" begin
    logger, seen = train(1)
    @test first.(seen) == [1, 2, 2, 1, 2, 2]
    @test logger.iterations == 1:4
    @test logger.losses == last.(seen[[1, 2, 4, 5]])

    logger, _ = train(2)
    @test logger.iterations == [2, 4]
end
