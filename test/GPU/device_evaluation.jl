using ForwardDiff: gradient, hessian
using Lux: Chain, Dense, WrappedFunction
using NeuralLyapunov
using NeuralPDE: GridTraining, PhysicsInformedNN, discretize
using Test

function device_network(f, structure, chain, weights, device)
    spec = NeuralLyapunovSpecification(
        structure, DontCheckNonnegativity(), AsymptoticStability()
    )
    pde = NeuralLyapunovPDESystem(
        f, [-2.0, -2.0], [2.0, 2.0], spec;
        state_syms = [:angle, :velocity], name = :device_evaluation
    )
    disc = PhysicsInformedNN(chain, GridTraining(1.0); init_params = vec(weights))
    prob = discretize(pde, disc)
    phi, params = get_network_and_parameters(prob, prob.u0)
    return phi, map(device, params)
end

function test_device_evaluation(device, on_device)
    fixed_point = zeros(2)
    states = [-1.0 0.0 0.5 1.0; 0.25 0.0 -0.5 1.0]

    @testset "Damped harmonic oscillator with device parameters" begin
        sho(x, p, t) = [x[2], -x[1] - x[2]]
        structure = NonnegativeStructure(2)
        phi, params = device_network(
            sho, structure, Dense(2 => 2; use_bias = false), [1.0 0.0; 0.0 1.0], device
        )
        @test all(on_device, values(params))
        @test on_device(phi[1](states, params.φ1))
        output = phi[1](view(states, :, 1:2), params.φ1)
        @test on_device(output)
        @test Array(output) == states[1:1, 1:2]
        @test Array(phi[1](device(states), params.φ1)) == states[1:1, :]

        for use_V̇_structure in (false, true)
            V, V̇ = get_numerical_lyapunov_function(
                phi, params, structure, sho, fixed_point; use_V̇_structure
            )
            @test vec(V(states)) ≈ vec(sum(abs2, states; dims = 1))
            @test vec(V̇(states)) ≈ -2 .* abs2.(states[2, :])
            @test V(fixed_point) == V̇(fixed_point) == 0
            @test gradient(V̇, fixed_point) == zeros(2)
            @test hessian(V̇, fixed_point) ≈ [0.0 0.0; 0.0 -4.0]
        end
    end

    return @testset "Inverted pendulum policy with device parameters" begin
        pendulum(x, u, p, t) = [x[2], sin(x[1]) + only(u)]
        function periodic_features(x)
            angle = x[1:1, :]
            return vcat(sin.(angle), 1 .- cos.(angle), x[2:2, :])
        end
        chain = Chain(WrappedFunction(periodic_features), Dense(3 => 4; use_bias = false))
        structure = add_policy_search(NonnegativeStructure(3), 1)
        weights = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0; -2.0 0.0 -1.0]
        phi, params = device_network(pendulum, structure, chain, weights, device)
        @test all(on_device, values(params))
        @test on_device(phi[4](states, params.φ4))

        periodic_states = states .+ [2π, 0.0]
        policy = get_policy(phi, params, structure; fixed_point)
        closed_loop(x, p, t) = pendulum(x, policy(x), p, t)
        @test vec(policy(states)) ≈ -2 .* sin.(states[1, :]) .- states[2, :]
        @test policy(periodic_states) ≈ policy(states)
        @test closed_loop(fixed_point, nothing, 0.0) == zeros(2)

        controlled = get_numerical_lyapunov_function(
            phi, params, structure, pendulum, fixed_point
        )
        structured = get_numerical_lyapunov_function(
            phi[1:3], params, NonnegativeStructure(3), closed_loop, fixed_point;
            use_V̇_structure = true
        )
        for (V, V̇) in (controlled, structured)
            @test vec(V(states)) ≈ 2 .* (1 .- cos.(states[1, :])) .+ abs2.(states[2, :])
            @test V(periodic_states) ≈ V(states)
            @test vec(V̇(states)) ≈ -2 .* abs2.(states[2, :])
            @test V(fixed_point) == V̇(fixed_point) == 0
            @test gradient(V, fixed_point) == gradient(V̇, fixed_point) == zeros(2)
            @test hessian(V, fixed_point) ≈ [2.0 0.0; 0.0 2.0]
            @test hessian(V̇, fixed_point) ≈ [0.0 0.0; 0.0 -4.0]
        end
    end
end
