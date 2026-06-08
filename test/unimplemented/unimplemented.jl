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

struct UnimplementedNeuralLyapunovStructure{nc} <: NeuralLyapunov.AbstractNeuralLyapunovStructure{nc} end

str = UnimplementedNeuralLyapunovStructure{true}()
@test_throws ErrorException NeuralLyapunov.get_control_structure(str)
@test_throws ErrorException NeuralLyapunov.get_control_dim(str)
@test_throws ErrorException NeuralLyapunov.get_V(str)
@test_throws ErrorException NeuralLyapunov.get_V̇(str)
@test_throws ErrorException NeuralLyapunov.get_network_dim(str)

str = UnimplementedNeuralLyapunovStructure{false}()
@test_throws ErrorException NeuralLyapunov.get_control_structure(str)
@test_throws ErrorException NeuralLyapunov.get_control_dim(str)
