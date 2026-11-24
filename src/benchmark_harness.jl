"""
    benchmark(dynamics, bounds, spec, chain, strategy, opt; <keyword_arguments>)
    benchmark(dynamics, lb, ub, spec, chain, strategy, opt; <keyword_arguments>)

Evaluate the specified neural Lyapunov method on the given system. Return a `NamedTuple`
containing the confusion matrix, optimization time, and other metrics listed below.

Train a neural Lyapunov function as specified, then discretize the domain using the provided
sampling algorithm and use the neural Lyapunov function to and the provided `classifier` to
predict whether grid points are in the region of attraction of the provided `fixed_point`.
Finally, simulate the system from each grid point and check if the trajectories reach the
fixed point. Return a confusion matrix for the neural Lyapunov classifier using the results
of the simulated trajectories as ground truth. Additionally return the time it took for the
optimization to run.

To train with multiple solvers, users should supply a vector of optimizers in `opt`. The
first optimizer will be used, then the problem will be remade with the result of the first
optimization as the initial guess. Then, the second optimizer will be used, and so on.
Supplying a vector of `Pair`s in `optimization_args` will use the same arguments for each
optimization pass, and supplying a vector of such vectors will use potentially different
arguments for each optimization pass.

Training runs on the CPU: NeuralPDE 7 does not support GPU training of a `PDESystem`
(see [NeuralPDE.jl#1161](https://github.com/SciML/NeuralPDE.jl/issues/1161)), so an
`init_params` stored on a GPU (or other accelerator) throws an `ArgumentError`. The
evaluation simulations can still run on the GPU by passing, e.g.,
[`EnsembleGPUArray`](https://docs.sciml.ai/DiffEqGPU/stable/manual/ensemblegpuarray/) as
`ensemble_alg`.

When `init_params` is not provided, it is generated using `Lux.initialparameters`. In that
case, the type of any
floating point parameters is inferred from `simulation_time`, the system parameters, and the
bounds, in that order. If any of those are not floats (e.g., integers), the next is
considered. If none are floats, `Float64` is used. This inference is important, since
simulation fails when the output type of the dynamics differs from the type of the state
divided by the type of `simulation_time`. For this reason, the `simulation_time` passed into
the ODE solver is converted to the same type as the network parameters (whether supplied by
the user or generated automatically).

# Positional Arguments
  - `dynamics`: the dynamical system being analyzed, represented as a `System` or the
    function `f` such that `ẋ = f(x[, u], p, t)`; either way, the ODE should not depend on
    time and only `t = 0.0` will be used. (For an example of when `f` would have a `u`
    argument, see [`add_policy_search`](@ref).) If `dynamics isa System`, call
    `mtkcompile(dynamics)` before `benchmark`, or
    `mtkcompile(dynamics; inputs = ..., split = false)` if the system has unbound inputs.
  - `bounds`: an array of domains, defining the training domain by bounding the states (and
    derivatives, when applicable) of `dynamics`; only used when `dynamics isa
    System`, otherwise use `lb` and `ub`.
  - `lb` and `ub`: the training domain will be ``[lb_1, ub_1]×[lb_2, ub_2]×...``; not used
    when `dynamics isa System`, then use `bounds`.
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
    `SciMLBase.NullParameters()`; not used when `dynamics isa System`, then use the
    default parameter values of `dynamics`.
  - `state_syms`: an array of the `Symbol` representing each state; not used when `dynamics
    isa System` (in that case, the symbols from `dynamics` are used); if `dynamics` is an
    `ODEFunction` or an `ODEInputFunction`, the symbols stored there are used, unless
    overridden here; if not provided here and cannot be inferred, `[:state1, :state2, ...]`
    will be used.
  - `parameter_syms`: an array of the `Symbol` representing each parameter; not used when
    `dynamics isa System` (in that case, the symbols from `dynamics` are used); if
    `dynamics` is an `ODEFunction` or an `ODEInputFunction`, the symbols stored there are
    used, unless overridden here; if not provided here and cannot be inferred,
    `[:param1, :param2, ...]` will be used.
  - `optimization_args`: arguments to be passed into the optimization solver, as a vector of
    `Pair`s. For more information, see the
    [Optimization.jl docs](https://docs.sciml.ai/Optimization/stable/API/solve/).
  - `log_frequency`: frequency (in iterations) at which to log the training loss; defaults
    to `50`. The loss is recorded by an Optimization.jl callback, which also draws new
    collocation points with `NeuralPDE.resample!` at every iteration (a no-op for training
    strategies that do not resample) and is called before any `callback` supplied in
    `optimization_args`.
  - `simulation_time`: simulation end time for checking if trajectory from a point reaches
    equilibrium
  - `ode_solver`: differential equation solver used in simulating the system for evaluation.
    For more information, see the
    [DifferentialEquations.jl docs](https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/).
  - `ode_solver_args`: arguments to be passed into the differential equation solver. For
    more information, see the
    [DifferentialEquations.jl docs](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/).
  - `ensemble_alg`: controls how the evaluation simulations are handled; defaults to
    `EnsembleDistributed()`, which uses `pmap` internally; see the
    [DifferentialEquations.jl docs](https://docs.sciml.ai/DiffEqDocs/stable/features/ensemble/)
    for more information.
  - `endpoint_check`: function of the endpoint of a simulation that returns `true` when the
    endpoint is approximately the fixed point and `false` otherwise; defaults to
    `(x) -> ≈(x, fixed_point; atol=atol)`.
  - `atol`: absolute tolerance used in the default value for `endpoint_check`.
  - `init_params`: initial parameters for the neural network, stored on the CPU; defaults
    to `nothing`, in which case the initial parameters are generated using
    `Lux.initialparameters` and `rng`.
  - `rng`: random number generator used to generate initial parameters, as well as in the
    default sampling algorithm; defaults to a `StableRNG` with seed `0`.

# Output Fields
  - `confusion_matrix`: confusion matrix of the neural Lyapunov classifier.
  - `data`: a `DataFrame` containing the following columns:
    - "Initial State": initial state of the simulation, split into separate columns for each
      state variable.
    - "Final State": end state of the simulation, split into separate columns for each state
      variable.
    - "V": value of the Lyapunov function at the initial state.
    - "dVdt": value of the Lyapunov decrease function at the initial state.
    - "Predicted in RoA": whether `classifier` predicted that the initial state is in the
      region of attraction.
    - "Actually in RoA": whether the endpoint of the simulation is approximately equal to
      `fixed_point` (as determined by `endpoint_check`).
    - "Classification": classification of each point, either "TP" (true positive),
      "TN" (true negative), "FP" (false positive), or "FN" (false negative).
  - `training_time`: time taken to train the neural Lyapunov function (in seconds).
  - `evaluation_time`: time taken to run the evaluation simulations (in seconds).
  - `θ`: the parameters of the neural Lyapunov function, as returned by
    [`get_network_and_parameters`](@ref).
  - `phi`: the neural network, as a `Vector` with one entry per neural network output (to
    be used with [`get_numerical_lyapunov_function`](@ref)), as returned by
    [`get_network_and_parameters`](@ref).
  - `V`: the neural Lyapunov function.
  - `V̇`: the Lyapunov decrease function.
  - `training_losses`: a `DataFrame` containing the training losses logged during training,
    with columns:
    - "Iteration": training iteration at which the loss was logged: the iteration
      reported by the optimizer (`state.iter` in the Optimization.jl callback), counted
      continuously across optimization passes when `opt` is a vector.
    - "Loss": the full weighted training loss at that iteration.
"""
function benchmark(
        dynamics::System,
        bounds,
        spec,
        chain,
        strategy,
        opt;
        n,
        fixed_point = nothing,
        optimization_args = [],
        simulation_time,
        ode_solver = AutoTsit5(Rosenbrock23()),
        ode_solver_args = [],
        atol = 1.0e-6,
        endpoint_check = nothing,
        classifier = (V, V̇, x) -> V̇ < zero(V̇),
        init_params = nothing,
        rng = StableRNG(0),
        sample_alg = LatinHypercubeSample(rng),
        ensemble_alg = EnsembleDistributed(),
        log_frequency = 50
    )
    params = setdiff(parameters(dynamics), unbound_inputs(dynamics))
    f = if isempty(unbound_inputs(dynamics))
        ODEFunction(dynamics)
    else
        ODEInputFunction(dynamics)
    end

    ics = initial_conditions(dynamics)
    p = [Symbolics.value(ics[param]) for param in params]

    bound_vars = map(b -> diff2term(b.variables), bounds)
    _bounds = map(unknowns(dynamics)) do x
        dom = bounds[findfirst(Base.Fix1(===, x), bound_vars)].domain
        (dom.left, dom.right)
    end
    lb = first.(_bounds)
    ub = last.(_bounds)

    default_float_type = if simulation_time isa AbstractFloat
        typeof(simulation_time)
    elseif eltype(p) <: AbstractFloat
        eltype(p)
    elseif eltype(ub) <: AbstractFloat
        eltype(ub)
    elseif eltype(lb) <: AbstractFloat
        eltype(lb)
    else
        Float64
    end

    type_changer = if default_float_type === Float64
        f64
    elseif default_float_type === Float32
        f32
    elseif default_float_type === Float16
        f16
    else
        identity
    end

    init_params = if init_params === nothing
        get_init_params(chain, rng) |> type_changer
    else
        init_params
    end

    fixed_point = if fixed_point === nothing
        zeros(default_float_type, length(bounds))
    else
        fixed_point
    end

    endpoint_check = if endpoint_check === nothing
        (x) -> ≈(x, fixed_point; atol = atol)
    else
        endpoint_check
    end

    logger = NeuralLyapunovBenchmarkLogger{default_float_type}()

    @named pde_system = NeuralLyapunovPDESystem(
        dynamics,
        bounds,
        spec;
        fixed_point
    )

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
        ensemble_alg,
        log_frequency,
        logger
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
        fixed_point = nothing,
        p = SciMLBase.NullParameters(),
        state_syms = [],
        parameter_syms = [],
        optimization_args = [],
        simulation_time,
        ode_solver = AutoTsit5(Rosenbrock23()),
        ode_solver_args = [],
        atol = 1.0e-6,
        endpoint_check = nothing,
        init_params = nothing,
        rng = StableRNG(0),
        sample_alg = LatinHypercubeSample(rng),
        ensemble_alg = EnsembleDistributed(),
        log_frequency = 50
    )
    default_float_type = if simulation_time isa AbstractFloat
        typeof(simulation_time)
    elseif eltype(p) <: AbstractFloat
        eltype(p)
    elseif eltype(ub) <: AbstractFloat
        eltype(ub)
    elseif eltype(lb) <: AbstractFloat
        eltype(lb)
    else
        Float64
    end

    type_changer = if default_float_type === Float64
        f64
    elseif default_float_type === Float32
        f32
    elseif default_float_type === Float16
        f16
    else
        identity
    end

    init_params = if init_params === nothing
        get_init_params(chain, rng) |> type_changer
    else
        init_params
    end

    fixed_point = if fixed_point === nothing
        zeros(default_float_type, length(lb))
    else
        fixed_point
    end

    endpoint_check = if endpoint_check === nothing
        (x) -> ≈(x, fixed_point; atol = atol)
    else
        endpoint_check
    end

    logger = NeuralLyapunovBenchmarkLogger{default_float_type}()

    @named pde_system = NeuralLyapunovPDESystem(
        dynamics,
        lb,
        ub,
        spec;
        fixed_point,
        p,
        state_syms,
        parameter_syms,
    )

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
        ensemble_alg,
        log_frequency,
        logger
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
        ensemble_alg,
        log_frequency,
        logger
    )
    check_host_params(init_params)

    # Construct OptimizationProblem
    discretization = PhysicsInformedNN(chain, strategy; init_params)
    opt_prob = discretize(pde_system, discretization)

    # Solve OptimizationProblem
    log_loss = LossLoggingCallback(logger, log_frequency)
    md = pinn_metadata(opt_prob)
    training_callback = function (state, loss)
        log_loss(state, loss)
        resample!(state.p, md)
        return false
    end
    optimization_args = with_training_callback(optimization_args, training_callback)
    training = @timed benchmark_solve(opt_prob, opt, optimization_args, log_loss)
    training_time = training.time

    # Get parameters from optimization result
    phi, θ = get_network_and_parameters(opt_prob, training.value)
    θ = θ |> cpud
    T = eltype(first(θ))

    V, V̇ = get_numerical_lyapunov_function(
        phi,
        θ,
        spec.structure,
        f,
        fixed_point;
        p
    )

    if neural_controller(spec.structure)
        f = let _f = f, u = get_policy(phi, θ, spec.structure; fixed_point)
            ODEFunction((x, _p, t) -> _f(x, u(x), _p, t))
        end
    end

    # Sample Lyapunov function and decrease function
    states = sample(n, T.(lb), T.(ub), sample_alg)
    V_samples = vec(V(states))
    V̇_samples = vec(V̇(states))

    (; endpoints, actual, predicted, evaluation_time) = simulate_ensemble(
        eachcol(states),
        V_samples,
        V̇_samples,
        f;
        classifier,
        simulation_time = T(simulation_time),
        p,
        ode_solver,
        ode_solver_args,
        endpoint_check,
        ensemble_alg
    )

    classification = Vector{String}(undef, length(actual))
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for (i, (a, p)) in enumerate(zip(actual, predicted))
        if a && p
            classification[i] = "TP"
            tp += 1
        elseif !a && p
            classification[i] = "FP"
            fp += 1
        elseif !a && !p
            classification[i] = "TN"
            tn += 1
        else
            classification[i] = "FN"
            fn += 1
        end
    end

    confusion_matrix = DataFrame(
        "Classification" => [
            "True Positives", "False Positives", "True Negatives", "False Negatives",
        ],
        "Count" => [tp, fp, tn, fn]
    )

    state_vars = string.(Symbol.(independent_variables(pde_system)))

    data = DataFrame(
        vcat(
            [
                "Initial " * var => map(x -> x[i], eachcol(states))
                    for (i, var) in enumerate(state_vars)
            ],
            [
                "Final " * var => map(x -> x[i], endpoints)
                    for (i, var) in enumerate(state_vars)
            ],
            [
                "V" => V_samples,
                "dVdt" => V̇_samples,
                "Predicted in RoA" => predicted,
                "Actually in RoA" => actual,
                "Classification" => classification,
            ]
        )
    )

    training_losses = DataFrame("Iteration" => logger.iterations, "Loss" => logger.losses)

    return (;
        confusion_matrix, data, training_time, evaluation_time, θ, phi, V, V̇,
        training_losses,
    )
end

function benchmark_solve(prob, opt, optimization_args, log_loss)
    # Solve OptimizationProblem
    start_pass!(log_loss)
    return solve(prob, opt; optimization_args...)
end

function benchmark_solve(
        prob,
        opt::AbstractVector,
        optimization_args::AbstractVector{<:AbstractVector},
        log_loss
    )
    # Solve OptimizationProblem
    res = Ref{Any}()
    for (_opt, args) in zip(opt, optimization_args)
        start_pass!(log_loss)
        _res = solve(prob, _opt; args...)
        prob = remake(prob, u0 = _trained_parameters(_res))
        res[] = _res
    end

    return res[]
end

function benchmark_solve(prob, opt::AbstractVector, optimization_args, log_loss)
    # Solve OptimizationProblem
    res = Ref{Any}()
    for _opt in opt
        start_pass!(log_loss)
        _res = solve(prob, _opt; optimization_args...)
        prob = remake(prob, u0 = _trained_parameters(_res))
        res[] = _res
    end

    return res[]
end

function simulate_ensemble(
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

    evaluation = @timed begin
        x0 = first(states)
        ensemble_prob = EnsembleProblem(
            ODEProblem(dynamics, x0, simulation_time, p);
            prob_func = (prob, ctx) -> remake(prob, u0 = states[ctx.sim_id]),
            output_func = (sol, ctx) -> (sol.u[end], false),
            u_init = fill(zeros(eltype(x0), size(x0)), length(states)),
            reduction = function (u, data, I)
                u[I] = data
                return u, false
            end
        )

        endpoints = solve(
            ensemble_prob,
            ode_solver,
            ensemble_alg;
            trajectories = length(states),
            ode_solver_args...
        ).u
    end
    evaluation_time = evaluation.time
    endpoints = evaluation.value

    actual = endpoint_check.(endpoints)

    return (; endpoints, actual, predicted, evaluation_time)
end

function check_host_params(init_params)
    device = get_device_type(init_params)
    device === Nothing || device <: CPUDevice || throw(
        ArgumentError(
            "`init_params` is stored on a $(device), but NeuralPDE 7 only supports \
            training a `PDESystem` on the CPU \
            (https://github.com/SciML/NeuralPDE.jl/issues/1161). Move `init_params` to \
            the CPU, e.g., with `cpu_device()(init_params)`."
        )
    )
    return nothing
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
