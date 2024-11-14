"""
    benchmark(dynamics, bounds, spec, chain, strategy, opt; <keyword_arguments>)
    benchmark(dynamics, lb, ub, spec, chain, strategy, opt; <keyword_arguments>)

Evaluate the specified neural Lyapunov method on the given system. Return the confusion
matrix and optimization time.

Train a neural Lyapunov function as specified, then discretize the domain using a grid
discretization and use the neural Lyapnov function to and the provided `classifier` to
predict whether grid points are in the region of attraction of the provided `fixed_point`.
Finally, simulate the system from each grid point and check if the trajectories reach the
fixed point. Return a confusion matrix for the neural Lyapunov classifier using the results
of the simulated trajectories as ground truth. Additionally return the time it took for the
optimization to run.

To use multiple solvers, users should supply a vector of optimizers in `opt`. The first
optimizer will be used, then the problem will be remade with the result of the first
optimization as the initial guess. Then, the second optimizer will be used, and so on.
Supplying a vector of `Pair`s in `optimization_args` will use the same arguments for each
optimization pass, and supplying a vector of such vectors will use potentially different
arguments for each optimization pass.

# Positional Arguments
  - `dynamics`: the dynamical system being analyzed, represented as an `ODESystem` or the
    function `f` such that `ẋ = f(x[, u], p, t)`; either way, the ODE should not depend on
    time and only `t = 0.0` will be used. (For an example of when `f` would have a `u`
    argument, see [`add_policy_search`](@ref).)
  - `bounds`: an array of domains, defining the training domain by bounding the states (and
    derivatives, when applicable) of `dynamics`; only used when `dynamics isa
    ODESystem`, otherwise use `lb` and `ub`.
  - `lb` and `ub`: the training domain will be ``[lb_1, ub_1]×[lb_2, ub_2]×...``; not used
    when `dynamics isa ODESystem`, then use `bounds`.
  - `spec`: a [`NeuralLyapunovSpecification`](@ref) defining the Lyapunov function
    structure, as well as the minimization and decrease conditions.
  - `chain`: a vector of Lux/Flux chains with a d-dimensional input and a 1-dimensional
    output corresponding to each of the dependent variables, where d is the length of
    `bounds` or `lb` and `ub`. Note that this specification respects the order of the
    dependent variables as specified in the PDESystem. Flux chains will be converted to Lux
    internally by NeuralPDE using `NeuralPDE.adapt(FromFluxAdaptor(false, false), chain)`.
  - `strategy`: determines which training strategy will be used. See the NeuralPDE Training
    Strategy documentation for more details.
  - `opt`: optimizer to use in training the neural Lyapunov function.

# Keyword Arguments
  - `n_grid`: number or grid points in each dimension used for evaluating the neural
    Lyapunov classifier.
  - `classifier`: function of ``V(x)``, ``V̇(x)``, and ``x`` that predicts whether ``x`` is
    in the region of attraction; defaults to `(V, V̇, x) -> V̇ < 0 || endpoint_check(x)`.
  - `fixed_point`: the equilibrium being analyzed; defaults to the origin.
  - `p`: the values of the parameters of the dynamical system being analyzed; defaults to
    `SciMLBase.NullParameters()`; not used when `dynamics isa ODESystem`, then use the
    default parameter values of `dynamics`.
  - `state_syms`: an array of the `Symbol` representing each state; not used when `dynamics
    isa ODESystem`, then the symbols from `dynamics` are used; if `dynamics isa ODEFunction`,
    symbols stored there are used, unless overridden here; if not provided here and cannot
    be inferred, `[:state1, :state2, ...]` will be used.
  - `parameter_syms`: an array of the `Symbol` representing each parameter; not used when
    `dynamics isa ODESystem`, then the symbols from `dynamics` are used; if `dynamics isa
    ODEFunction`, symbols stored there are used, unless overridden here; if not provided
    here and cannot be inferred, `[:param1, :param2, ...]` will be used.
  - `policy_search::Bool`: whether or not to include a loss term enforcing `fixed_point` to
    actually be a fixed point; defaults to `false`; only used when `dynamics isa Function &&
    !(dynamics isa ODEFunction)`; when `dynamics isa ODEFunction`, `policy_search` should
    not be supplied (as it must be false); when `dynamics isa ODESystem`, value inferred by
    the presence of unbound inputs.
  - `optimization_args`: arguments to be passed into the optimization solver, as a vector of
    `Pair`s. For more information, see the
    [Optimization.jl docs](https://docs.sciml.ai/Optimization/stable/API/solve/).
  - `simulation_time`: simulation end time for checking if trajectory from a point reaches
    equilibrium
  - `ode_solver`: differential equation solver used in simulating the system for evaluation.
    For more information, see the
    [DifferentialEquations.jl docs](https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/).
  - `ode_solver_args`: arguments to be passed into the differential equation solver. For
    more information, see the
    [DifferentialEquations.jl docs](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/).
  - `endpoint_check`: function of the endpoint of a simulation that returns `true` when the
    endpoint is approximately the fixed point and `false` otherwise; defaults to
    `(x) -> ≈(x, fixed_point; atol=atol)`.
  - `atol`: absolute tolerance used in the default value for `endpoint_check`.
  - `verbose`: enable verbose output. Instead of outputting
    `(confusion_matrix, training_time)`, output
    `((confusion_matrix, training_time), (states, endpoints, actual, predicted, V_samples, V̇_samples))`,
    where `states` is the grid of evaluation points, `endpoints` is the endpoints of the
    simulations, `actual` is the result of `endpoint_check` applied to `endpoints`,
    `predicted` is the result of `classifier` applied to `states`, `V_samples` is ``V``
    evaluated at `states`, and `V̇_samples` is ``V̇`` evaluated at `states`.
"""
function benchmark(
    dynamics::ODESystem,
    bounds,
    spec,
    chain,
    strategy,
    opt;
    n_grid,
    fixed_point = zeros(length(bounds)),
    optimization_args = [],
    simulation_time,
    ode_solver = Tsit5(),
    ode_solver_args = [],
    atol = 1e-6,
    endpoint_check = (x) -> ≈(x, fixed_point; atol=atol),
    classifier = (V, V̇, x) -> V̇ < zero(V̇) || endpoint_check(x),
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
        n_grid,
        classifier,
        fixed_point,
        p,
        optimization_args,
        simulation_time,
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
    n_grid,
    classifier = (V, V̇, x) -> V̇ < zero(V̇),
    fixed_point = zeros(length(lb)),
    p = SciMLBase.NullParameters(),
    state_syms = [],
    parameter_syms = [],
    policy_search = false,
    optimization_args = [],
    simulation_time,
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
        n_grid,
        classifier,
        fixed_point,
        p,
        optimization_args,
        simulation_time,
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
    n_grid,
    classifier,
    fixed_point,
    p,
    optimization_args,
    simulation_time,
    ode_solver,
    ode_solver_args,
    endpoint_check,
    verbose
)
    t = @timed benchmark_solve(
        pde_system,
        chain,
        strategy,
        opt,
        optimization_args
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

function benchmark_solve(pde_system, chain, strategy, opt, optimization_args)
    # Construct OptimizationProblem
    discretization = PhysicsInformedNN(chain, strategy)
    prob = discretize(pde_system, discretization)

    # Solve OptimizationProblem
    res = solve(prob, opt; optimization_args...)

    # Return parameters θ and network phi
    return res.u.depvar, discretization.phi
end

function benchmark_solve(pde_system, chain, strategy, opt::AbstractVector, optimization_args::AbstractVector{<:AbstractVector})
    # Construct OptimizationProblem
    discretization = PhysicsInformedNN(chain, strategy)
    prob = discretize(pde_system, discretization)

    # Solve OptimizationProblem
    res = Ref{Any}()
    for i in eachindex(opt)
        _res = solve(prob, opt[i]; optimization_args[i]...)
        prob = remake(prob, u0 = _res.u)
        res[] = _res
    end

    # Return parameters θ and network phi
    return res[].u.depvar, discretization.phi
end

function benchmark_solve(pde_system, chain, strategy, opt::AbstractVector, optimization_args)
    # Construct OptimizationProblem
    discretization = PhysicsInformedNN(chain, strategy)
    prob = discretize(pde_system, discretization)

    # Solve OptimizationProblem
    res = for i in eachindex(opt)
        _res = solve(prob, opt[i]; optimization_args...)
        prob = Optimization.remake(prob, u0 = _res.u)
        _res
    end

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
