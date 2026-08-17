using NeuralLyapunov
using LinearAlgebra: ⋅
using Test

struct GenericStructure <: NeuralLyapunov.AbstractNeuralLyapunovStructure{false}
    network_dim::Int
end

NeuralLyapunov.get_V(::GenericStructure) = (phi, state, fixed_point) -> phi(state)[1]
NeuralLyapunov.get_V̇(::GenericStructure) =
    (phi, J_phi, state, dstate_dt, fixed_point) -> J_phi(state)[1, :] ⋅ dstate_dt
NeuralLyapunov.get_network_dim(s::GenericStructure) = s.network_dim

struct GenericControlStructure <: NeuralLyapunov.AbstractNeuralLyapunovStructure{true}
    network_dim::Int
    control_dim::Int
end

NeuralLyapunov.get_V(::GenericControlStructure) = (phi, state, fixed_point) -> phi(state)[1]
NeuralLyapunov.get_V̇(::GenericControlStructure) =
    (phi, J_phi, state, dstate_dt, fixed_point) -> J_phi(state)[1, :] ⋅ dstate_dt
NeuralLyapunov.get_network_dim(s::GenericControlStructure) = s.network_dim
NeuralLyapunov.get_control_dim(s::GenericControlStructure) = s.control_dim
NeuralLyapunov.get_control_structure(::GenericControlStructure) =
    (phi_c, state, fixed_point) -> phi_c(state)

struct GenericMinimizationCondition <: NeuralLyapunov.AbstractLyapunovMinimizationCondition end

NeuralLyapunov.check_nonnegativity(::GenericMinimizationCondition) = true
NeuralLyapunov.check_minimal_fixed_point(::GenericMinimizationCondition) = true
NeuralLyapunov.get_minimization_condition(::GenericMinimizationCondition) =
    (V, state, fixed_point) -> V(state) - V(fixed_point)

struct GenericDecreaseCondition <: NeuralLyapunov.AbstractLyapunovDecreaseCondition end

NeuralLyapunov.check_decrease(::GenericDecreaseCondition) = true
NeuralLyapunov.get_decrease_condition(::GenericDecreaseCondition) =
    (V, V̇, state, fixed_point) -> V̇(state)

@testset "Structure interface" begin
    state = [2.0, 3.0]
    fixed_point = zeros(2)
    phi = x -> [x[1], x[2]^2]
    J_phi = x -> [1.0 0.0; 0.0 2x[2]]

    structure = GenericStructure(2)
    @test NeuralLyapunov.get_V(structure)(phi, state, fixed_point) == 2.0
    @test NeuralLyapunov.get_V̇(structure)(phi, J_phi, state, [4.0, 5.0], fixed_point) == 4.0
    @test NeuralLyapunov.get_network_dim(structure) == 2
    @test !NeuralLyapunov.neural_controller(structure)

    control_structure = GenericControlStructure(3, 1)
    @test NeuralLyapunov.get_network_dim(control_structure) == 3
    @test NeuralLyapunov.get_control_dim(control_structure) == 1
    @test NeuralLyapunov.get_control_structure(control_structure)(phi, state, fixed_point) ==
        [2.0, 9.0]
    @test NeuralLyapunov.neural_controller(control_structure)

    augmented = NeuralLyapunov.add_policy_search(structure, 1)
    @test NeuralLyapunov.neural_controller(augmented)
    @test NeuralLyapunov.get_network_dim(augmented) == 3
    @test NeuralLyapunov.get_control_dim(augmented) == 1
end

@testset "Condition interfaces" begin
    V = x -> sum(abs2, x)
    V̇ = x -> -sum(abs2, x)
    state = [2.0, 3.0]
    fixed_point = zeros(2)

    minimization = GenericMinimizationCondition()
    @test NeuralLyapunov.check_nonnegativity(minimization)
    @test NeuralLyapunov.check_minimal_fixed_point(minimization)
    @test NeuralLyapunov.get_minimization_condition(minimization)(V, state, fixed_point) ==
        13.0

    decrease = GenericDecreaseCondition()
    @test NeuralLyapunov.check_decrease(decrease)
    @test NeuralLyapunov.get_decrease_condition(decrease)(V, V̇, state, fixed_point) == -13.0
end
