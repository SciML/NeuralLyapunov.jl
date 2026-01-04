using NeuralLyapunov
using Test

struct UnimplementedMinimizationCondition <:
    NeuralLyapunov.AbstractLyapunovMinimizationCondition end

cond = UnimplementedMinimizationCondition()

@test_throws ErrorException NeuralLyapunov.check_nonnegativity(cond)
@test_throws ErrorException NeuralLyapunov.check_minimal_fixed_point(cond)
@test_throws ErrorException NeuralLyapunov.get_minimization_condition(cond)

struct UnimplementedDecreaseCondition <: NeuralLyapunov.AbstractLyapunovDecreaseCondition end

cond = UnimplementedDecreaseCondition()

@test_throws ErrorException NeuralLyapunov.check_decrease(cond)
@test_throws ErrorException NeuralLyapunov.get_decrease_condition(cond)
