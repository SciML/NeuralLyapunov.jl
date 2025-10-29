using ModelingToolkit
import ModelingToolkit: inputs
using NeuralLyapunovProblemLibrary
using OrdinaryDiffEq
using Plots
using Test, StableRNGs

rng = StableRNG(0)

################## Undriven pendulum should drop to downward equilibrium ###################
@testset "Undriven pendulum test" begin
    println("Undriven pendulum test")

    @named pendulum_undriven = Pendulum(; driven = false)
    pendulum_undriven = mtkcompile(pendulum_undriven)

    x0 = π * rand(rng, 2)
    p = rand(rng, 2)
    τ = 1 / prod(p)
    prob = ODEProblem(
        pendulum_undriven,
        merge(
            Dict(unknowns(pendulum_undriven) .=> x0),
            Dict(parameters(pendulum_undriven) .=> p)
        ),
        15τ
    )
    sol = solve(prob, Tsit5())
    x_end, y_end, ω_end = sin(sol.u[end][1]), -cos(sol.u[end][1]), sol.u[end][2]
    @test sqrt(sum(abs2, [x_end, y_end] .- [0, -1]))≈0 atol=1e-4
    @test ω_end≈0 atol=1e-4

    anim = plot_pendulum(sol)
    @test anim isa Plots.Animation
    # gif(anim, fps=50)
end

############################# Feedback cancellation controller #############################
@testset "Simple pendulum feedback cancellation test" begin
    println("Simple pendulum feedback cancellation test")

    @named pendulum = Pendulum()

    π_cancellation(x, p) = 2 * p[2]^2 * sin(x[1])

    pendulum_simplified = mtkcompile(
        pendulum;
        inputs = inputs(pendulum),
        outputs = [],
        simplify = true,
        split = false
    )

    t, = independent_variables(pendulum)
    Dt = Differential(t)

    p = map(Base.Fix1(getproperty, pendulum), toexpr.(parameters(pendulum)))
    u = map(
        Base.Fix1(getproperty, pendulum),
        toexpr.(getproperty.(inputs(pendulum_simplified), :f))
    )

    @named cancellation_controller = System(
        u .~ π_cancellation([pendulum.θ, Dt(pendulum.θ)], p),
        t,
        u,
        p
    )
    @named pendulum_feedback_cancellation = compose(cancellation_controller, pendulum)
    pendulum_feedback_cancellation = mtkcompile(pendulum_feedback_cancellation)

    # Swing up to upward equilibrium
    x0 = rand(rng, 2)
    p = rand(rng, 2)
    τ = 1 / prod(p)
    prob = ODEProblem(
        pendulum_feedback_cancellation,
        merge(
            Dict(unknowns(pendulum_feedback_cancellation) .=> x0),
            Dict(parameters(pendulum_feedback_cancellation) .=> p)
        ),
        15τ
    )
    sol = solve(prob, Tsit5())
    x_end, y_end, ω_end = sin(sol.u[end][1]), -cos(sol.u[end][1]), sol.u[end][2]
    @test sqrt(sum(abs2, [x_end, y_end] .- [0, 1]))≈0 atol=1e-4
    @test ω_end≈0 atol=1e-4
end
