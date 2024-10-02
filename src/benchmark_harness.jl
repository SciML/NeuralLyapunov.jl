function benchmark(
    dynamics::ODESystem,
    bounds,
    spec,
    chain,
    strategy,
    opt;
    simulation_time,
    n_grid,
    classifier = (V, V̇, x) -> V̇ < zero(V̇),
    fixed_point = zeros(length(bounds)),
    optimization_args = [],
    ode_solver = Tsit5(),
    ode_solver_args = [],
    atol = 1e-6,
    endpoint_check = (x) -> ≈(x, fixed_point; atol=atol),
    verbose = false
)
    @named pde_system = NeuralLyapunovPDESystem(
        dynamics,
        bounds,
        spec;
        fixed_point = fixed_point
    )

    f, params = if isempty(ModelingToolkit.unbound_inputs(dynamics))
        ODEFunction(dynamics), parameters(dynamics)
    else
        (_f, _), _, _p = ModelingToolkit.generate_control_function(dynamics, simplify = true)
        _f, _p
    end

    defaults = ModelingToolkit.get_defaults(dynamics)
    p = [defaults[param] for param in params]

    lb = [d.domain.left for d in bounds]
    ub = [d.domain.right for d in bounds]

    return _benchmark(
        pde_system,
        f,
        lb,
        ub,
        spec,
        chain,
        strategy,
        opt;
        simulation_time,
        n_grid,
        classifier,
        fixed_point,
        p,
        optimization_args,
        ode_solver,
        ode_solver_args,
        endpoint_check,
        verbose
    )
end

function benchmark(
    dynamics::Function,
    lb,
    ub,
    spec,
    chain,
    strategy,
    opt;
    simulation_time,
    n_grid,
    classifier = (V, V̇, x) -> V̇ < zero(V̇),
    fixed_point = zeros(length(lb)),
    p = SciMLBase.NullParameters(),
    state_syms = [],
    parameter_syms = [],
    policy_search = false,
    optimization_args = [],
    ode_solver = Tsit5(),
    ode_solver_args = [],
    atol = 1e-6,
    endpoint_check = (x) -> ≈(x, fixed_point; atol=atol),
    verbose = false
)
    # Build PDESystem
    @named pde_system = NeuralLyapunovPDESystem(
        dynamics,
        lb,
        ub,
        spec;
        fixed_point = fixed_point,
        p = p,
        state_syms = state_syms,
        parameter_syms = parameter_syms,
        policy_search = policy_search
    )

    return _benchmark(
        pde_system,
        dynamics,
        lb,
        ub,
        spec,
        chain,
        strategy,
        opt;
        simulation_time,
        n_grid,
        classifier,
        fixed_point,
        p,
        optimization_args,
        ode_solver,
        ode_solver_args,
        endpoint_check,
        verbose
    )
end

function _benchmark(
    pde_system,
    f,
    lb,
    ub,
    spec,
    chain,
    strategy,
    opt;
    simulation_time,
    n_grid,
    classifier,
    fixed_point,
    p,
    optimization_args,
    ode_solver,
    ode_solver_args,
    endpoint_check,
    verbose
)
    t = @timed benchmark_solve(
        pde_system,
        chain,
        strategy,
        opt;
        optimization_args = optimization_args
    )
    solve_time = t.time
    θ, phi = t.value

    V, V̇ = get_numerical_lyapunov_function(
        phi,
        θ,
        spec.structure,
        f,
        fixed_point;
        p = p
    )

    f = let fc = spec.structure.f_call, _f = f,
            net = NeuralLyapunov.phi_to_net(phi, θ)
            ODEFunction((x, _p, t) -> fc(_f, net, x, _p, t))
        end

    states, V_samples, V̇_samples = eval_Lyapunov(lb, ub, n_grid, V, V̇)

    if verbose
        cm, states, endpoints, actual, predicted, V_samples, V̇_samples = build_confusion_matrix(
            collect(states),
            V_samples,
            V̇_samples,
            f;
            classifier,
            fixed_point,
            simulation_time,
            p,
            ode_solver,
            ode_solver_args,
            endpoint_check,
            verbose
        )
        return ((cm, solve_time), (states, endpoints, actual, predicted, V_samples, V̇_samples))
    else
        cm = build_confusion_matrix(
            collect(states),
            V_samples,
            V̇_samples,
            f;
            classifier,
            fixed_point,
            simulation_time,
            p,
            ode_solver,
            ode_solver_args,
            endpoint_check,
            verbose
        )
        return (cm, solve_time)
    end
end

function benchmark_solve(pde_system, chain, strategy, opt; optimization_args)
    # Construct OptimizationProblem
    discretization = PhysicsInformedNN(chain, strategy)
    prob = discretize(pde_system, discretization)

    # Solve OptimizationProblem
    res = solve(prob, opt; optimization_args...)

    # Return parameters θ and network phi
    return res.u.depvar, discretization.phi
end

function eval_Lyapunov(lb, ub, n, V, V̇)
    Δ = @. (ub - lb) / n

    ranges = collect(l:δ:u for (l, δ, u) in zip(lb, Δ, ub))
    states = Iterators.map(collect, Iterators.product(ranges...))

    V_samples = V.(states)
    V̇_samples = V̇.(states)

    return (states, V_samples, V̇_samples)
end

function get_endpoint(
    f::ODEFunction,
    x0,
    t_end;
    p,
    solver,
    solver_args
)
    prob = ODEProblem(f, x0, t_end, p)
    sol = solve(prob, solver; solver_args...)
    return sol.u[end]
end

function build_confusion_matrix(
    states,
    V_samples,
    V̇_samples,
    dynamics;
    classifier,
    fixed_point,
    simulation_time,
    p,
    ode_solver,
    ode_solver_args,
    endpoint_check,
    verbose
)
    predicted = classifier.(V_samples, V̇_samples, states)
    ix0 = findfirst(x -> x == fixed_point, states)
    if !isnothing(ix0)
        predicted[ix0] = true
    end

    endpoints = [
        get_endpoint(
            dynamics,
            x,
            simulation_time;
            p = p,
            solver = ode_solver,
            solver_args = ode_solver_args
        ) for x in states
    ]

    actual = endpoint_check.(endpoints)

    cm = ConfusionMatrix(vec(actual), vec(predicted))

    if verbose
        (cm, states, endpoints, actual, predicted, V_samples, V̇_samples)
    else
        cm
    end
end
