using NeuralLyapunov, NeuralLyapunovProblemLibrary, ModelingToolkit
using ModelingToolkit: unbound_inputs
using SciMLBase: SymbolCache, ODEInputFunction
using ForwardDiff
using LinearAlgebra: eigvals
using Test

###################### Damped simple harmonic oscillator ######################
@testset "Simple harmonic oscillator local Lyapunov" begin
    println("Local Lyapunov: Damped SHO")

    # Define dynamics
    "Simple Harmonic Oscillator Dynamics"
    function sho(state, p, t)
        ζ, ω_0 = p
        pos = state[1]
        vel = state[2]
        return vcat(vel, -2ζ * ω_0 * vel - ω_0^2 * pos)
    end
    p = [1.0, 1.0]
    fixed_point = [0.0, 0.0]

    V, V̇ = get_quadratic_lyapunov_function(sho; fixed_point, p)

    # Sample V and V̇ on a grid
    lb = [-5.0, -2.0]
    ub = [5.0, 2.0]
    Δx = (ub[1] - lb[1]) / 100
    Δv = (ub[2] - lb[2]) / 100
    xs = lb[1]:Δx:ub[1]
    vs = lb[2]:Δv:ub[2]
    states = Iterators.map(collect, Iterators.product(xs, vs))

    # Test positive definiteness of V
    V_samples = vec(V(reduce(hcat, filter(x -> sum(abs2, x) != 0, collect(states)))))
    @test minimum(V_samples) > 0
    @test V(fixed_point) == 0

    # Since sho is linear, V̇(x, v) = -x^2 - v^2 when Q is the identity matrix
    V̇_samples = vec(V̇(reduce(hcat, states)))
    V̇_expected = -vec(map(Base.Fix1(sum, abs2), states))
    @test maximum(abs, V̇_samples - V̇_expected) < 1.0e-10


    # Test again with ODEFunction
    sho_odef = ODEFunction(sho; sys = SymbolCache([:x, :v], [:ζ, :ω_0]))
    V, V̇ = get_quadratic_lyapunov_function(sho_odef; fixed_point, p)

    # Test positive definiteness of V
    V_samples = vec(V(reduce(hcat, filter(x -> sum(abs2, x) != 0, collect(states)))))
    @test minimum(V_samples) > 0
    @test V(fixed_point) == 0

    # Since sho is linear, V̇(x, v) = -x^2 - v^2 when Q is the identity matrix
    V̇_samples = vec(V̇(reduce(hcat, states)))
    V̇_expected = -vec(map(Base.Fix1(sum, abs2), states))
    @test maximum(abs, V̇_samples - V̇_expected) < 1.0e-10


    # Test again with explicit Jacobian
    function sho_jac(x, p, t)
        ζ, ω_0 = p
        return [zero(ζ) one(ζ); -ω_0^2 -2ζ * ω_0]
    end
    sho_odef = ODEFunction(sho; jac = sho_jac, sys = SymbolCache([:x, :v], [:ζ, :ω_0]))
    V, V̇ = get_quadratic_lyapunov_function(sho_odef; fixed_point, p)

    # Test positive definiteness of V
    V_samples = vec(V(reduce(hcat, filter(x -> sum(abs2, x) != 0, collect(states)))))
    @test minimum(V_samples) > 0
    @test V(fixed_point) == 0

    # Since sho is linear, V̇(x, v) = -x^2 - v^2 when Q is the identity matrix
    V̇_samples = vec(V̇(reduce(hcat, states)))
    V̇_expected = -vec(map(Base.Fix1(sum, abs2), states))
    @test maximum(abs, V̇_samples - V̇_expected) < 1.0e-10
end

############################ Inverted pendulum LQR ############################
@testset "LQR on inverted pendulum local Lyapunov" begin
    println("Local Lyapunov: Inverted Pendulum - LQR")

    # Define dynamics
    function open_loop_pendulum_dynamics(x, u, p, t)
        θ, ω = x
        ζ, ω_0 = p
        τ = u[]
        return [
            ω
            -2ζ * ω_0 * ω - ω_0^2 * sin(θ) + τ
        ]
    end
    upright_equilibrium = [π, 0.0]
    p = [0.5, 1.0]

    V, V̇ = get_quadratic_lyapunov_function(
        open_loop_pendulum_dynamics;
        fixed_point = upright_equilibrium,
        p,
        u_dim = 1
    )

    # Sample V and V̇ on a grid
    lb = [0.0, -2.0]
    ub = [2π, 2.0]
    Δθ = (ub[1] - lb[1]) / 100
    Δω = (ub[2] - lb[2]) / 100
    θs = lb[1]:Δθ:ub[1]
    ωs = lb[2]:Δω:ub[2]
    states = Iterators.map(collect, Iterators.product(θs, ωs))

    # Test positive definiteness of V
    V_samples = vec(V(reduce(hcat, filter(x -> sum(abs2, x) != 0, collect(states)))))
    @test minimum(V_samples) > 0
    @test V(upright_equilibrium) == 0

    # Test local negative definiteness of V̇
    @test V̇(upright_equilibrium) == 0
    @test maximum(abs, ForwardDiff.gradient(V̇, upright_equilibrium)) < 1.0e-10
    @test maximum(eigvals(ForwardDiff.hessian(V̇, upright_equilibrium))) < 0


    # Test again with ODEInputFunction
    state_syms = [:θ, :ω]
    parameter_syms = [:ζ, :ω_0]
    open_loop_pendulum_odeif = ODEInputFunction(
        open_loop_pendulum_dynamics;
        sys = SymbolCache(state_syms, parameter_syms)
    )

    V, V̇ = get_quadratic_lyapunov_function(
        open_loop_pendulum_odeif;
        fixed_point = upright_equilibrium,
        p,
        u_eq = [0.0]
    )

    # Test positive definiteness of V
    V_samples = vec(V(reduce(hcat, filter(x -> sum(abs2, x) != 0, collect(states)))))
    @test minimum(V_samples) > 0
    @test V(upright_equilibrium) == 0

    # Test local negative definiteness of V̇
    @test V̇(upright_equilibrium) == 0
    @test maximum(abs, ForwardDiff.gradient(V̇, upright_equilibrium)) < 1.0e-10
    @test maximum(eigvals(ForwardDiff.hessian(V̇, upright_equilibrium))) < 0


    # Test again with explicit Jacobian and control jacobian
    function open_loop_pendulum_jac(x, u, p, t)
        θ, ω = x
        ζ, ω_0 = p
        return [zero(ζ) one(ζ); -ω_0^2 * cos(θ) -2ζ * ω_0]
    end
    function open_loop_pendulum_control_jac(x, u, p, t)
        T = eltype(u)
        return [zero(T); one(T)]
    end
    open_loop_pendulum_odeif = ODEInputFunction(
        open_loop_pendulum_dynamics;
        jac = open_loop_pendulum_jac,
        controljac = open_loop_pendulum_control_jac,
        sys = SymbolCache(state_syms, parameter_syms)
    )

    V, V̇ = get_quadratic_lyapunov_function(
        open_loop_pendulum_odeif;
        fixed_point = upright_equilibrium,
        p,
        u_eq = [0.0]
    )

    # Test positive definiteness of V
    V_samples = vec(V(reduce(hcat, filter(x -> sum(abs2, x) != 0, collect(states)))))
    @test minimum(V_samples) > 0
    @test V(upright_equilibrium) == 0

    # Test local negative definiteness of V̇
    @test V̇(upright_equilibrium) == 0
    @test maximum(abs, ForwardDiff.gradient(V̇, upright_equilibrium)) < 1.0e-10
    @test maximum(eigvals(ForwardDiff.hessian(V̇, upright_equilibrium))) < 0
end

############################### Damped pendulum ###############################
@testset "Damped pendulum (ODESystem) local Lyapunov" begin
    println("Local Lyapunov: Damped Pendulum (ODESystem)")

    # Define dynamics
    @mtkcompile damped_pendulum = Pendulum(; driven = false, defaults = Float32[5.0, 1.0])
    fixed_point = zeros(2)

    V, V̇ = get_quadratic_lyapunov_function(damped_pendulum)

    # Test positive definiteness of V
    lb = [-π, -2.0]
    ub = [π, 2.0]
    Δθ = (ub[1] - lb[1]) / 100
    Δω = (ub[2] - lb[2]) / 100
    θs = lb[1]:Δθ:ub[1]
    ωs = lb[2]:Δω:ub[2]
    states = Iterators.map(collect, Iterators.product(θs, ωs))
    V_samples = vec(V(reduce(hcat, filter(x -> sum(abs2, x) != 0, collect(states)))))
    @test minimum(V_samples) > 0
    @test V(fixed_point) == 0

    # Test local negative definiteness of V̇
    @test V̇(fixed_point) == 0
    @test maximum(abs, ForwardDiff.gradient(V̇, fixed_point)) < 1.0e-10
    @test maximum(eigvals(ForwardDiff.hessian(V̇, fixed_point))) < 0
end

############################ Planar quadrotor LQR #############################
@testset "LQR on planar quadrotor (ODESystem) local Lyapunov" begin
    println("Local Lyapunov: Planar Quadrotor - LQR (ODESystem)")

    # Define dynamics
    # Assume rotors are negligible mass when calculating the moment of inertia
    m, r = ones(2)
    g = 1.0
    I_quad = m * r^2 / 12
    p = [m, I_quad, g, r]
    @named quadrotor = QuadrotorPlanar(; defaults = p)
    u1, u2 = unbound_inputs(quadrotor)
    quadrotor = mtkcompile(quadrotor; inputs = [u1, u2], split = false)

    u_eq = fill(m * g / 2, 2)
    fixed_point = zeros(6)

    V, V̇ = get_quadratic_lyapunov_function(quadrotor; u_eq)

    # Test local negative definiteness of V̇
    @test V̇(fixed_point) == 0
    @test maximum(abs, ForwardDiff.gradient(V̇, fixed_point)) < 1.0e-10
    @test maximum(eigvals(ForwardDiff.hessian(V̇, fixed_point))) < 0
end
