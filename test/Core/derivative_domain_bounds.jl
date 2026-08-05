using ModelingToolkit, NeuralLyapunov, NeuralLyapunovProblemLibrary
using ModelingToolkit: t_nounits as t, D_nounits as D
using Test

println("Derivative-specified domain bounds")

######################### Define dynamics and domain ##########################

@mtkcompile damped_pendulum = Pendulum(; driven = false, defaults = [0.5, 1.0])
θ, ω = unknowns(damped_pendulum)

# `mtkcompile` names the velocity state `θˍt(t)`, so a user may equivalently write the
# bound on it as `D(θ)`; both spellings must reach the same domain variable.
bounds_by_state = [θ ∈ (-π, π), ω ∈ (-10.0, 10.0)]
bounds_by_derivative = [θ ∈ (-π, π), D(θ) ∈ (-10.0, 10.0)]

####################### Specify neural Lyapunov problem #######################

structure = PositiveSemiDefiniteStructure(3)
minimization_condition = DontCheckNonnegativity(check_fixed_point = false)
decrease_condition = ExponentialStability(0.5)
spec = NeuralLyapunovSpecification(structure, minimization_condition, decrease_condition)

############################# Construct PDESystem #############################

@named pde_by_state = NeuralLyapunovPDESystem(damped_pendulum, bounds_by_state, spec)
@named pde_by_derivative = NeuralLyapunovPDESystem(
    damped_pendulum, bounds_by_derivative, spec
)

@testset "Derivative-specified domain bounds" begin
    @test isequal(pde_by_derivative.domain, pde_by_state.domain)
    @test Set(Symbol.(map(d -> d.variables, pde_by_state.domain))) == Set([:θ, :θˍt])

    # Bounds may be listed in any order; matching is by variable, not position.
    @named pde_reordered = NeuralLyapunovPDESystem(
        damped_pendulum, reverse(bounds_by_derivative), spec
    )
    @test Set(map(d -> (d.variables, d.domain), pde_reordered.domain)) ==
        Set(map(d -> (d.variables, d.domain), pde_by_state.domain))
end
