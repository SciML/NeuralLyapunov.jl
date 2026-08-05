"""
    NeuralLyapunovPDESystem(dynamics::System, bounds, spec; <keyword_arguments>)
    NeuralLyapunovPDESystem(dynamics, lb, ub, spec; <keyword_arguments>)

Construct a `ModelingToolkit.PDESystem` representing the specified neural Lyapunov problem.

# Positional Arguments
  - `dynamics`: the dynamical system being analyzed, represented as a `System` or the
    function `f` such that `ẋ = f(x[, u], p, t)`; either way, the ODE should not depend on
    time and only `t = 0` will be used. (For an example of when `f` would have a `u`
    argument, see [`add_policy_search`](@ref).) If `dynamics isa System`, call
    `mtkcompile(dynamics)` before `NeuralLyapunovPDESystem`, or
    `mtkcompile(dynamics; inputs = ..., split = false)` if the system has unbound inputs.
  - `bounds`: an array of domains, defining the training domain by bounding the states (and
    derivatives, when applicable) of `dynamics`; only used when `dynamics isa
    System`, otherwise use `lb` and `ub`.
  - `lb` and `ub`: the training domain will be ``[lb_1, ub_1]×[lb_2, ub_2]×...``; not used
    when `dynamics isa System`, then use `bounds`.
  - `spec`: a [`NeuralLyapunovSpecification`](@ref) defining the Lyapunov function
    structure, as well as the minimization and decrease conditions.

# Keyword Arguments
  - `fixed_point`: the equilibrium being analyzed; defaults to the origin.
  - `p`: the values of the parameters of the dynamical system being analyzed; defaults to
    `SciMLBase.NullParameters()`; not used when `dynamics isa System`, then use the
    default parameter values of `dynamics`.
  - `state_syms`: an array of the `Symbol` representing each state; not used when `dynamics
    isa System` (in that case, the symbols from `dynamics` are used); if `dynamics` is an
    `ODEFunction` or an `ODEInputFunction`, the symbols stored there are used, unless
    overridden here; if not provided here and cannot be inferred, `[:x1, :x2, ...]` will be
    used.
  - `parameter_syms`: an array of the `Symbol` representing each parameter; not used when
    `dynamics isa System` (in that case, the symbols from `dynamics` are used); if
    `dynamics` is an `ODEFunction` or an `ODEInputFunction`, the symbols stored there are
    used, unless overridden here; if not provided here and cannot be inferred,
    `[:p1, :p2, ...]` will be used.
  - `policy_search::Bool`: whether or not to include a loss term enforcing `fixed_point` to
    actually be a fixed point; defaults to `false`; when `dynamics isa System`, the value
    is inferred by the presence of unbound inputs and when `dynamics` is an `ODEFunction` or
    an `ODEInputFunction`, the value is inferred by the type of `dynamics`.
  - `name`: the name of the constructed `PDESystem`.
"""
function NeuralLyapunovPDESystem(
        dynamics,
        lb,
        ub,
        spec::NeuralLyapunovSpecification;
        fixed_point = zeros(length(lb)),
        p = SciMLBase.NullParameters(),
        state_syms = [],
        parameter_syms = [],
        name
    )::PDESystem
    ########################## Define state symbols ###########################
    state_dim = length(lb)

    # Define state symbols, if not already defined
    if isempty(state_syms)
        state_syms = [Symbol(:x, i) for i in 1:state_dim]
    end
    state = [first(@parameters $s) for s in state_syms]

    ######################## Define parameter symbols #########################
    # Define parameter symbols, if not already defined
    if p == SciMLBase.NullParameters()
        if !isempty(parameter_syms)
            error(
                "Got nonempty parameter_syms when p == SciMLBase.NullParameters(). " *
                    "Please provide p or leave parameter_syms empty."
            )
        end
    elseif isempty(parameter_syms)
        parameter_syms = [Symbol(:p, i) for i in 1:length(p)]
    elseif length(parameter_syms) != length(p)
        error(
            "Length of parameter_syms ($(length(parameter_syms))) and of p ($(length(p)))" *
                " do not match."
        )
    end

    params = [first(@parameters $s) for s in parameter_syms]

    ##################### Define default parameter values #####################
    initial_conditions = if p == SciMLBase.NullParameters()
        Dict()
    else
        Dict([param => param_val for (param, param_val) in zip(params, p)])
    end

    ############################# Define domains ##############################
    domains = [state[i] in (lb[i], ub[i]) for i in 1:state_dim]

    ######################### Check for policy search #########################
    policy_search = neural_controller(spec.structure)

    ########################### Construct PDESystem ###########################
    return _NeuralLyapunovPDESystem(
        dynamics,
        domains,
        spec,
        fixed_point,
        state,
        params,
        initial_conditions,
        policy_search,
        name
    )
end

function NeuralLyapunovPDESystem(
        dynamics::Union{ODEFunction, ODEInputFunction},
        lb,
        ub,
        spec::NeuralLyapunovSpecification;
        fixed_point = zeros(length(lb)),
        p = SciMLBase.NullParameters(),
        state_syms = [],
        parameter_syms = [],
        name
    )::PDESystem
    if dynamics.mass_matrix !== I
        throw(
            ErrorException(
                "DAEs are not supported at this time. Please supply dynamics" *
                    " without a mass matrix."
            )
        )
    end

    policy_search = neural_controller(spec.structure)
    if policy_search && (dynamics isa ODEFunction)
        throw(
            ErrorException(
                "Got spec.structure isa AbstractNeuralLyapunovStructure{true} when " *
                    "dynamics were supplied as an ODEFunction f(x,p,t)."
            )
        )
    elseif !policy_search && (dynamics isa ODEInputFunction)
        throw(
            ErrorException(
                "Got spec.structure isa AbstractNeuralLyapunovStructure{false} when " *
                    "dynamics were supplied as an ODEInputFunction f(x,u,p,t)."
            )
        )
    end

    # Extract state and parameter symbols from ODEFunction/ODEInputFunction
    s_syms, p_syms = if dynamics.sys isa System
        s_syms = Symbol.(operation.(unknowns(dynamics.sys)))
        p_syms = Symbol.(parameters(dynamics.sys))
        (s_syms, p_syms)
    elseif dynamics.sys isa SymbolCache
        s_syms = variable_symbols(dynamics.sys)
        p_syms = if isnothing(dynamics.sys.parameters)
            []
        else
            keys(dynamics.sys.parameters)
        end
        (s_syms, p_syms)
    else
        ([], [])
    end

    # Override ODEFunction/ODEInputFunction state and parameter symbols when supplied
    if !isempty(state_syms)
        s_syms = state_syms
    end
    if !isempty(parameter_syms)
        p_syms = parameter_syms
    end

    return NeuralLyapunovPDESystem(
        dynamics.f,
        lb,
        ub,
        spec;
        fixed_point,
        p,
        state_syms = s_syms,
        parameter_syms = p_syms,
        name
    )
end

function NeuralLyapunovPDESystem(
        dynamics::System,
        bounds,
        spec::NeuralLyapunovSpecification;
        fixed_point = zeros(length(bounds)),
        name
    )::PDESystem
    ######################### Check for policy search #########################
    policy_search = !isempty(unbound_inputs(dynamics))
    if policy_search && !neural_controller(spec.structure)
        throw(
            ErrorException(
                "Got unbound inputs in dynamics but spec.structure isa " *
                    "AbstractNeuralLyapunovStructure{false}."
            )
        )
    elseif !policy_search && neural_controller(spec.structure)
        throw(
            ErrorException(
                "Got spec.structure isa AbstractNeuralLyapunovStructure{true} but no " *
                    "unbound inputs in dynamics."
            )
        )
    end

    f = if policy_search
        ODEInputFunction(dynamics)
    else
        ODEFunction(dynamics)
    end

    ########################## Define state symbols ###########################
    # States should all be functions of time, but we just want the symbol
    # e.g., if the state is ω(t), we just want ω
    _state = operation.(unknowns(dynamics))
    state_syms = Symbol.(_state)
    state = [first(@parameters $s) for s in state_syms]

    ###################### Remove derivatives in domains ######################
    domains = map(d -> operation(diff2term(d.variables)) ∈ d.domain, bounds)
    domain_vars = map(d -> d.variables, domains)
    if Set(_state) != Set(domain_vars)
        error(
            "Domain variables from `domains` do not match those extracted from " *
                "`dynamics`. Got $_state from `dynamics` and $domain_vars from `domains`."
        )
    end

    ########################### Construct PDESystem ###########################
    return _NeuralLyapunovPDESystem(
        f.f,
        domains,
        spec,
        fixed_point,
        state,
        setdiff(parameters(dynamics), unbound_inputs(dynamics)),
        initial_conditions(dynamics),
        policy_search,
        name
    )
end

function _NeuralLyapunovPDESystem(
        dynamics,
        domains,
        spec::NeuralLyapunovSpecification,
        fixed_point,
        state,
        params,
        initial_conditions,
        policy_search::Bool,
        name
    )::PDESystem
    ########################## Unpack specifications ##########################
    structure = spec.structure
    minimization_condition = spec.minimization_condition
    decrease_condition = spec.decrease_condition

    ################## Define Lyapunov function & derivative ##################
    output_dim = get_network_dim(structure)
    control_dim = policy_search ? get_control_dim(structure) : 0
    net_syms = [Symbol(:φ, i) for i in 1:output_dim]
    net = [first(@variables $s(..)) for s in net_syms]

    # φ(x) is the symbolic form of neural network output
    φ(x) = Num.([φi(x...) for φi in net[1:(output_dim - control_dim)]])

    # u(x) is the symbolic form of the control input, when applicable
    if policy_search
        φu(x) = Num.([φi(x...) for φi in net[(output_dim - control_dim + 1):end]])
        _u = get_control_structure(structure)
        u(x) = _u(φu, x, fixed_point)
    end

    # V(x) is the symbolic form of the Lyapunov function
    _V = get_V(structure)
    V(x) = _V(φ, x, fixed_point)

    # V̇(x) is the symbolic time derivative of the Lyapunov function
    if policy_search
        f = x -> dynamics(x, u(x), params, 0)
    else
        f = x -> dynamics(x, params, 0)
    end
    _V̇ = get_V̇(structure)
    Jφ(x) = Symbolics.jacobian(φ(x), x)
    V̇(x) = _V̇(φ, Jφ, x, f(x), fixed_point)


    ################ Define equations and boundary conditions #################
    eqs = Equation[]

    if check_nonnegativity(minimization_condition)
        cond = get_minimization_condition(minimization_condition)::Function
        cond_eq = cond(V, state, fixed_point) .~ 0
        if cond_eq isa Equation
            push!(eqs, cond_eq)
        elseif cond_eq isa AbstractVector{Equation}
            append!(eqs, cond_eq)
        else
            error(
                "Minimization condition function must return an Equation or vector of ",
                "Equations. Instead got $(typeof(cond_eq))."
            )
        end
    end

    if check_decrease(decrease_condition)
        cond = get_decrease_condition(decrease_condition)::Function
        cond_eq = cond(V, V̇, state, fixed_point) .~ 0
        if cond_eq isa Equation
            push!(eqs, cond_eq)
        elseif cond_eq isa AbstractVector{Equation}
            append!(eqs, cond_eq)
        else
            error(
                "Decrease condition function must return an Equation or vector of ",
                "Equations. Instead got $(typeof(cond_eq))."
            )
        end
    end

    bcs = Equation[]

    if check_minimal_fixed_point(minimization_condition)
        V0 = V(fixed_point)
        V0 = V0 isa AbstractVector ? V0[] : V0
        push!(bcs, V0 ~ 0)
    end

    if policy_search
        append!(bcs, f(fixed_point) .~ 0)
    end

    if isempty(eqs) && isempty(bcs)
        error("No training conditions specified.")
    end

    # NeuralPDE requires an equation and a boundary condition, even if they are trivial
    # like φ(0) ~ φ(0), so we remove any trivial equations if they showed up naturally
    # alongside other equations and add some in if we have no other equations
    eqs = filter(eq -> eq != (0 ~ 0), eqs)
    bcs = filter(eq -> eq != (0 ~ 0), bcs)

    if isempty(eqs)
        push!(eqs, φ(fixed_point)[1] ~ φ(fixed_point)[1])
    end
    if isempty(bcs)
        push!(bcs, φ(fixed_point)[1] ~ φ(fixed_point)[1])
    end

    ########################### Construct PDESystem ###########################
    dvs = policy_search ? vcat(φ(state), φu(state)) : φ(state)
    return PDESystem(
        eqs,
        bcs,
        domains,
        state,
        dvs,
        params;
        initial_conditions,
        name
    )
end
