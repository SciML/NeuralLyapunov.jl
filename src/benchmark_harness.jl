"""
    benchmark(dynamics, bounds, spec, chain, strategy, opt; <keyword_arguments>)
    benchmark(dynamics, lb, ub, spec, chain, strategy, opt; <keyword_arguments>)

Evaluate the specified neural Lyapunov method on the given system. Return a `NamedTuple`
containing the confusion matrix, optimization time, and other metrics listed below.

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
  - `n`: number of samples used for evaluating the neural Lyapunov classifier.
  - `sample_alg`: sampling algorithm used for generating the evaluation data; defaults to
    `LatinHypercubeSample(rng)`; see the
    [QuasiMonteCarlo.jl docs](https://docs.sciml.ai/QuasiMonteCarlo/stable/samplers/) for
    more information.
  - `classifier`: function of ``V(x)``, ``V̇(x)``, and ``x`` that predicts whether ``x`` is
    in the region of attraction; when constructing the confusion matrix, a point is
    predicted to be in the region of attraction if `classifier` or `endpoint_check` returns
    `true`; defaults to `(V, V̇, x) -> V̇ < 0`.
  - `fixed_point`: the equilibrium being analyzed; defaults to the origin.
  - `p`: the values of the parameters of the dynamical system being analyzed; defaults to
    `SciMLBase.NullParameters()`; not used when `dynamics isa ODESystem`, then use the
    default parameter values of `dynamics`.
  - `state_syms`: an array of the `Symbol` representing each state; not used when `dynamics
    isa ODESystem` (in that case, the symbols from `dynamics` are used); if `dynamics` is an
    `ODEFunction` or an `ODEInputFunction`, the symbols stored there are used, unless
    overridden here; if not provided here and cannot be inferred, `[:state1, :state2, ...]`
    will be used.
  - `parameter_syms`: an array of the `Symbol` representing each parameter; not used when
    `dynamics isa ODESystem` (in that case, the symbols from `dynamics` are used); if
    `dynamics` is an `ODEFunction` or an `ODEInputFunction`, the symbols stored there are
    used, unless overridden here; if not provided here and cannot be inferred,
    `[:param1, :param2, ...]` will be used.
  - `policy_search::Bool`: whether or not to include a loss term enforcing `fixed_point` to
    actually be a fixed point; defaults to `false`; when `dynamics isa ODESystem`, the value
    is inferred by the presence of unbound inputs and when `dynamics` is an `ODEFunction` or
    an `ODEInputFunction`, the value is inferred by the type of `dynamics`.
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
  - `ensemble_alg`: controls how the evaluation simulations are handled; defaults to
    `EnsembleThreads()`, which uses multithreading (local parallelism only); see the
    [DifferentialEquations.jl docs](https://docs.sciml.ai/DiffEqDocs/stable/features/ensemble/)
    for more information.
  - `endpoint_check`: function of the endpoint of a simulation that returns `true` when the
    endpoint is approximately the fixed point and `false` otherwise; defaults to
    `(x) -> ≈(x, fixed_point; atol=atol)`.
  - `atol`: absolute tolerance used in the default value for `endpoint_check`.
  - `init_params`: initial parameters for the neural network; defaults to `nothing`, in
    which case the initial parameters are generated using `Lux.initialparameters` and `rng`.
  - `init_states`: initial states for the neural network; defaults to `nothing`, in which
    case the initial states are generated using `Lux.initialstates` and `rng`.
  - `rng`: random number generator used to generate initial parameters and states, as well
    as in the default sampling algorithm; defaults to a `StableRNG` with seed `0`.

# Output Fields
  - `data, training_time, θ, phi, V, V̇`
  - `confusion_matrix`: confusion matrix of the neural Lyapunov classifier.
  - `data`: a `DataFrame` containing the following columns:
    - "Initial state": initial state of the simulation.
    - "End state": end state of the simulation.
    - "V": value of the Lyapunov function at the initial state.
    - "dVdt": value of the Lyapunov decrease function at the initial state.
    - "Predicted in RoA": whether `classifier` predicted that the initial state is in the
      region of attraction.
    - "Actually in RoA": whether the endpoint of the simulation is approximately equal to
      `fixed_point` (as determined by `endpoint_check`).
    - "Classification": classification of each point, either "TP" (true positive),
      "TN" (true negative), "FP" (false positive), or "FN" (false negative).
  - `training_time`: time taken to train the neural Lyapunov function.
  - `θ`: the parameters of the neural Lyapunov function.
  - `phi`: the neural network, represented as `phi(x, θ)` if the neural network has a single
    output, or a `Vector` of the same with one entry per neural network output (to be used
    with [`get_numerical_lyapunov_function`](@ref)).
  - `V`: the neural Lyapunov function.
  - `V̇`: the Lyapunov decrease function.
"""
function benchmark(
        dynamics::ODESystem,
        bounds,
        spec,
        chain,
        strategy,
        opt;
        n,
        fixed_point = zeros(length(bounds)),
        optimization_args = [],
        simulation_time,
        ode_solver = Tsit5(),
        ode_solver_args = [],
        atol = 1e-6,
        endpoint_check = (x) -> ≈(x, fixed_point; atol = atol),
        classifier = (V, V̇, x) -> V̇ < zero(V̇),
        init_params = nothing,
        init_states = nothing,
        rng = StableRNG(0),
        sample_alg = LatinHypercubeSample(rng),
        ensemble_alg = EnsembleThreads()
)
    @named pde_system = NeuralLyapunovPDESystem(
        dynamics,
        bounds,
        spec;
        fixed_point
    )

    params = parameters(dynamics)
    f = if isempty(unbound_inputs(dynamics))
        ODEFunction(dynamics)
    else
        dynamics_io_sys,
        _ = structural_simplify(
            dynamics, (unbound_inputs(dynamics), []); split = false)
        ODEInputFunction(dynamics_io_sys; simplify = true, split = false)
    end

    defs = defaults(dynamics)
    p = [defs[param] for param in params]

    lb = [d.domain.left for d in bounds]
    ub = [d.domain.right for d in bounds]

    init_params = if init_params === nothing
        get_init_params(chain, rng)
    else
        init_params
    end

    init_states = if init_states === nothing
        get_init_states(chain, rng)
    else
        init_states
    end

    _classifier(V, V̇, x) = classifier(V, V̇, x) || endpoint_check(x)

    return _benchmark(
        pde_system,
        f,
        lb,
        ub,
        spec,
        chain,
        strategy,
        opt;
        n,
        sample_alg,
        classifier = _classifier,
        fixed_point,
        p,
        optimization_args,
        simulation_time,
        ode_solver,
        ode_solver_args,
        endpoint_check,
        init_params,
        init_states,
        ensemble_alg
    )
end

function benchmark(
        dynamics,
        lb,
        ub,
        spec,
        chain,
        strategy,
        opt;
        n,
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
        endpoint_check = (x) -> ≈(x, fixed_point; atol = atol),
        init_params = nothing,
        init_states = nothing,
        rng = StableRNG(0),
        sample_alg = LatinHypercubeSample(rng),
        ensemble_alg = EnsembleThreads()
)
    # Build PDESystem
    @named pde_system = NeuralLyapunovPDESystem(
        dynamics,
        lb,
        ub,
        spec;
        fixed_point,
        p,
        state_syms,
        parameter_syms,
        policy_search
    )

    init_params = if init_params === nothing
        get_init_params(chain, rng)
    else
        init_params
    end

    init_states = if init_states === nothing
        get_init_states(chain, rng)
    else
        init_states
    end

    _classifier(V, V̇, x) = classifier(V, V̇, x) || endpoint_check(x)

    return _benchmark(
        pde_system,
        dynamics,
        lb,
        ub,
        spec,
        chain,
        strategy,
        opt;
        n,
        sample_alg,
        classifier = _classifier,
        fixed_point,
        p,
        optimization_args,
        simulation_time,
        ode_solver,
        ode_solver_args,
        endpoint_check,
        init_params,
        init_states,
        ensemble_alg
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
        n,
        sample_alg,
        classifier,
        fixed_point,
        p,
        optimization_args,
        simulation_time,
        ode_solver,
        ode_solver_args,
        endpoint_check,
        init_params,
        init_states,
        ensemble_alg
)
    t = @timed begin
        # Construct OptimizationProblem
        discretization = PhysicsInformedNN(chain, strategy; init_params, init_states)
        opt_prob = discretize(pde_system, discretization)

        # Solve OptimizationProblem
        u = benchmark_solve(opt_prob, opt, optimization_args)

        # Get parameters from optimization result
        phi = discretization.phi
        θ = phi isa AbstractArray ? u.depvar : u
        (θ, phi)
    end
    training_time = t.time
    θ, phi = t.value

    V, V̇ = get_numerical_lyapunov_function(
        phi,
        θ,
        spec.structure,
        f,
        fixed_point;
        p
    )

    f = let fc = spec.structure.f_call, _f = f, net = phi_to_net(phi, θ)
        ODEFunction((x, _p, t) -> fc(_f, net, x, _p, t))
    end

    # Sample Lyapunov function and decrease function
    cpud = cpu_device()
    states = sample(n, lb, ub, sample_alg) |> cpud
    V_samples = V(states) |> cpud
    V̇_samples = V̇(states) |> cpud

    (; confusion_matrix, endpoints, actual, predicted) = build_confusion_matrix(
        eachcol(states),
        first.(eachcol(V_samples)),
        first.(eachcol(V̇_samples)),
        f;
        classifier,
        simulation_time,
        p,
        ode_solver,
        ode_solver_args,
        endpoint_check,
        ensemble_alg
    )

    classification = map(zip(actual, predicted)) do (a, p)
        (a == p ? "T" : "F") * (p ? "P" : "N")
    end

    data = DataFrame(
        "Initial state" => eachcol(states),
        "End state" => endpoints,
        "V" => first.(eachcol(V_samples)),
        "dVdt" => first.(eachcol(V̇_samples)),
        "Predicted in RoA" => predicted,
        "Actually in RoA" => actual,
        "Classification" => classification
    )

    return (; confusion_matrix, data, training_time, θ, phi, V, V̇)
end

function benchmark_solve(prob, opt, optimization_args)
    # Solve OptimizationProblem
    res = solve(prob, opt; optimization_args...)

    # Return optimization result
    return res.u
end

function benchmark_solve(
        prob,
        opt::AbstractVector,
        optimization_args::AbstractVector{<:AbstractVector}
)
    # Solve OptimizationProblem
    res = Ref{Any}()
    for (_opt, args) in zip(opt, optimization_args)
        _res = solve(prob, _opt; args...)
        prob = remake(prob, u0 = _res.u)
        res[] = _res
    end

    # Return optimization result
    return res[].u
end

function benchmark_solve(prob, opt::AbstractVector, optimization_args)
    # Solve OptimizationProblem
    res = Ref{Any}()
    for _opt in opt
        _res = solve(prob, _opt; optimization_args...)
        prob = remake(prob, u0 = _res.u)
        res[] = _res
    end

    # Return optimization result
    return res[].u
end

function build_confusion_matrix(
        states,
        V_samples,
        V̇_samples,
        dynamics;
        classifier,
        simulation_time,
        p,
        ode_solver,
        ode_solver_args,
        ensemble_alg,
        endpoint_check
)
    predicted = classifier.(V_samples, V̇_samples, states)

    ensemble_prob = EnsembleProblem(
        ODEProblem(dynamics, first(states), simulation_time, p);
        prob_func = (prob, i, repeat) -> remake(prob, u0 = states[i]),
        output_func = (sol, i) -> (sol.u[end], false)
    )
    endpoints = solve(
        ensemble_prob,
        ode_solver,
        ensemble_alg;
        trajectories = length(states),
        ode_solver_args...
    ).u

    actual = endpoint_check.(endpoints)

    confusion_matrix = ConfusionMatrix(vec(actual), vec(predicted))

    return (; confusion_matrix, endpoints, actual, predicted)
end

function get_init_params(chain, rng)
    return if chain isa AbstractArray
        map(chain) do c
            LuxCore.initialparameters(rng, c)
        end
    else
        LuxCore.initialparameters(rng, chain)
    end
end

function get_init_states(chain, rng)
    return if chain isa AbstractArray
        map(chain) do c
            LuxCore.initialstates(rng, c)
        end
    else
        LuxCore.initialstates(rng, chain)
    end
end
