using ModelingToolkit
import ModelingToolkit: inputs
using NeuralLyapunovProblemLibrary
using OrdinaryDiffEq
using Plots
using LinearAlgebra
using ControlSystemsBase: lqr, Continuous
using Test, StableRNGs

rng = StableRNG(0)

#################################### Hovering quadrotor ####################################
@testset "Planar quadrotor vertical only test" begin
    println("Planar quadrotor vertical only test")

    @named quadrotor_planar = QuadrotorPlanar()

    function π_vertical_only(x, p; y_goal = 0.0, k_p = 1.0, k_d = 1.0)
        y, ẏ = x[2], x[5]
        m, I_quad, g, r = p
        T0 = m * g / 2
        T = T0 - k_p * m * g / r * (y - y_goal) - k_d * m * sqrt(g / r) * ẏ
        return [T, T]
    end

    quadrotor_planar_simplified = mtkcompile(
        quadrotor_planar;
        inputs = inputs(quadrotor_planar),
        outputs = [],
        simplify = true,
        split = false
    )

    t, = independent_variables(quadrotor_planar)
    Dt = Differential(t)
    q = setdiff(unknowns(quadrotor_planar), inputs(quadrotor_planar))

    params = map(
        Base.Fix1(getproperty, quadrotor_planar), toexpr.(parameters(quadrotor_planar)))
    u = map(
        Base.Fix1(getproperty, quadrotor_planar),
        toexpr.(getproperty.(inputs(quadrotor_planar_simplified), :f))
    )
    q = map(
        Base.Fix1(getproperty, quadrotor_planar),
        toexpr.(getproperty.(q, :f))
    )
    x = vcat(q, Dt.(q))

    @named vertical_only_controller = System(
        u .~ π_vertical_only(x, params),
        t,
        vcat(x, u),
        params
    )

    @named quadrotor_planar_vertical_only = compose(vertical_only_controller, quadrotor_planar)
    quadrotor_planar_vertical_only = mtkcompile(quadrotor_planar_vertical_only)

    # Hovering
    # Assume rotors are negligible mass when calculating the moment of inertia
    x0 = Dict(x .=> zeros(6))
    x0[q[2]] = rand(rng)
    x0[x[5]] = rand(rng)
    m, r = ones(2)
    g = 1.0
    I_quad = m * r^2 / 12
    p = Dict(params .=> [m, I_quad, g, r])
    τ = sqrt(r / g)

    prob = ODEProblem(quadrotor_planar_vertical_only, merge(x0, p), 15τ)
    sol = solve(prob, Tsit5())

    x_end, y_end, θ_end = sol[q][end]
    v_x_end, v_y_end, v_θ_end = sol[Dt.(q)][end]
    @test x_end≈0.0 atol=1e-4
    @test y_end≈0.0 atol=1e-4
    @test θ_end≈0.0 atol=1e-4
    @test v_x_end≈0.0 atol=1e-4
    @test v_y_end≈0.0 atol=1e-4
    @test v_θ_end≈0.0 atol=1e-4

    anim = plot_quadrotor_planar(
        sol,
        [m, I_quad, g, r];
        x_symbol = q[1],
        y_symbol = q[2],
        θ_symbol = q[3],
        u1_symbol = u[1],
        u2_symbol = u[2]
    )
    @test anim isa Plots.Animation
    # gif(anim, fps = 50)
end

############################## LQR planar quadrotor controller #############################
@testset "Planar quadrotor LQR test" begin
    println("Planar quadrotor LQR test")

    function quadrotor_planar_lqr_matrix(p; Q = I(6), R = I(2))
        m, I_quad, g, r = p

        # Assumes linearization around a fixed point
        # x_eq = (x*, y*, 0, 0, 0, 0), u_eq = (mg / 2, mg / 2)
        A_lin = zeros(6, 6)
        A_lin[1:3, 4:6] .= I(3)
        A_lin[4, 3] = -g

        B_lin = zeros(6, 2)
        B_lin[5, :] .= 1 / m
        B_lin[6, :] .= r / I_quad, -r / I_quad

        return lqr(Continuous, A_lin, B_lin, Q, R)
    end

    function π_lqr(p; x_eq = zeros(6), Q = I(6), R = I(2))
        L = quadrotor_planar_lqr_matrix(p; Q, R)
        m, _, g, _ = p
        T0 = m * g / 2
        return (x) -> -L * (x - x_eq) + [T0, T0]
    end

    @named quadrotor_planar = QuadrotorPlanar()

    quadrotor_planar_simplified = mtkcompile(
        quadrotor_planar;
        inputs = inputs(quadrotor_planar),
        outputs = [],
        simplify = true,
        split = false
    )

    t, = independent_variables(quadrotor_planar)
    Dt = Differential(t)
    q = setdiff(unknowns(quadrotor_planar), inputs(quadrotor_planar))
    params = map(
        Base.Fix1(getproperty, quadrotor_planar), toexpr.(parameters(quadrotor_planar)))
    u = map(
        Base.Fix1(getproperty, quadrotor_planar),
        toexpr.(getproperty.(inputs(quadrotor_planar_simplified), :f))
    )
    q = map(
        Base.Fix1(getproperty, quadrotor_planar),
        toexpr.(getproperty.(q, :f))
    )
    x = vcat(q, Dt.(q))

    # Assume rotors are negligible mass when calculating the moment of inertia
    m, r = ones(2)
    g = 1.0
    I_quad = m * r^2 / 12
    p = [m, I_quad, g, r]

    @named lqr_controller = System(
        u .~ π_lqr(p)(x),
        t,
        vcat(x, u),
        params
    )

    @named quadrotor_planar_lqr = compose(lqr_controller, quadrotor_planar)
    quadrotor_planar_lqr = mtkcompile(quadrotor_planar_lqr)

    # Fly to origin
    x0 = Dict(x .=> 0.02 * rand(rng, 6) .- 0.01)
    p = Dict(params .=> [m, I_quad, g, r])
    τ = sqrt(r / g)

    prob = ODEProblem(quadrotor_planar_lqr, merge(x0, p), 15τ)
    sol = solve(prob, Tsit5())

    x_end, y_end, θ_end = sol[q][end]
    v_x_end, v_y_end, v_θ_end = sol[Dt.(q)][end]
    @test x_end≈0.0 atol=1e-4
    @test y_end≈0.0 atol=1e-4
    @test θ_end≈0.0 atol=1e-4
    @test v_x_end≈0.0 atol=1e-4
    @test v_y_end≈0.0 atol=1e-4
    @test v_θ_end≈0.0 atol=1e-4

    anim = plot_quadrotor_planar(
        sol,
        [m, I_quad, g, r];
        x_symbol = q[1],
        y_symbol = q[2],
        θ_symbol = q[3],
        u1_symbol = u[1],
        u2_symbol = u[2]
    )
    @test anim isa Plots.Animation
    # gif(anim, fps = 50)
end
