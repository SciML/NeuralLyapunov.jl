"""
    NeuralLyapunovPDESystem(dynamics::ODESystem, bounds, spec; <keyword_arguments>)
    NeuralLyapunovPDESystem(dynamics::Function, lb, ub, spec; <keyword_arguments>)

Construct and return a `PDESystem` representing the specified neural Lyapunov problem, along
with a function representing the neural network.

The returned neural network function takes three inputs: the neural network structure `phi`,
the trained network parameters, and a matrix of inputs, then operates columnwise on the
inputs.

# Arguments
- `dynamics`: the dynamical system being analyzed, represented as an `ODESystem` or the
        function `f` such that `ẋ = f(x[, u], p, t)`; either way, the ODE should not depend
        on time and only `t = 0.0` will be used
- `bounds`: an array of domains, defining the training domain by bounding the states (and
        derivatives, when applicable) of `dynamics`; only used when `dynamics isa
        ODESystem`, otherwise use `lb` and `ub`.
- `lb` and `ub`: the training domain will be ``[lb_1, ub_1]×[lb_2, ub_2]×...``; not used
        when `dynamics isa ODESystem`, then use `bounds`.
- `spec::NeuralLyapunovSpecification`: defines the Lyapunov function structure, as well as
        the minimization and decrease conditions.
- `fixed_point`: the equilibrium being analyzed; defaults to the origin.
- `p`: the values of the parameters of the dynamical system being analyzed; defaults to
        `SciMLBase.NullParameters()`; not used when `dynamics isa ODESystem`, then use the
        default parameter values of `dynamics`.
- `state_syms`: an array of the `Symbol` representing each state; not used when `dynamics
        isa ODESystem`, then the symbols from `dynamics` are used; if `dynamics isa
        ODEFunction`, symbols stored there are used, unless overridden here; if not provided
        here and cannot be inferred, `[:state1, :state2, ...]` will be used.
- `parameter_syms`: an array of the `Symbol` representing each parameter; not used when
        `dynamics isa ODESystem`, then the symbols from `dynamics` are used; if `dynamics
        isa ODEFunction`, symbols stored there are used, unless overridden here; if not
        provided here and cannot be inferred, `[:param1, :param2, ...]` will be used.
- `policy_search::Bool`: whether or not to include a loss term enforcing `fixed_point` to
        actually be a fixed point; defaults to `false`; only used when `dynamics isa
        Function && !(dynamics isa ODEFunction)`; when `dynamics isa ODEFunction`,
        `policy_search` must be `false`, so should not be supplied; when `dynamics isa
        ODESystem`, value inferred by the presence of unbound inputs.
"""
function NeuralLyapunovPDESystem(
        dynamics::Function,
        lb,
        ub,
        spec::NeuralLyapunovSpecification;
        fixed_point = zeros(length(lb)),
        p = SciMLBase.NullParameters(),
        state_syms = [],
        parameter_syms = [],
        policy_search::Bool = false
)::Tuple{PDESystem, Function}
    ########################## Define state symbols ###########################
    state_dim = length(lb)

    # Define state symbols, if not already defined
    state_syms = if isempty(state_syms)
        [Symbol(:state, i) for i in 1:state_dim]
    else
        state_syms
    end
    state = [first(@parameters $s) for s in state_syms]

    ######################## Define parameter symbols #########################
    # Define parameter symbols, if not already defined
    param_syms = if p == SciMLBase.NullParameters()
        []
    else
        if isempty(parameter_syms)
            [Symbol(:param, i) for i in 1:length(p)]
        else
            parameter_syms
        end
    end

    params = [first(@parameters $s) for s in param_syms]

    ##################### Define default parameter values #####################
    defaults = if p == SciMLBase.NullParameters()
        Dict()
    else
        Dict([param => param_val for (param, param_val) in zip(params, p)])
    end

    ############################# Define domains ##############################
    domains = [state[i] ∈ (lb[i], ub[i]) for i in 1:state_dim]

    ########################### Construct PDESystem ###########################
    return _NeuralLyapunovPDESystem(
        dynamics,
        domains,
        spec,
        fixed_point,
        state,
        params,
        defaults,
        policy_search
    )
end

function NeuralLyapunovPDESystem(
        dynamics::ODEFunction,
        lb,
        ub,
        spec::NeuralLyapunovSpecification;
        fixed_point = zeros(length(lb)),
        p = SciMLBase.NullParameters()
)::Tuple{PDESystem, Function}
    if dynamics.mass_matrix !== I
        throw(ErrorException("DAEs are not supported at this time. Please supply dynamics" *
                             " without a mass matrix."))
    end

    s_syms, p_syms = if dynamics.sys isa ODESystem
        s_syms = Symbol.(operation.(states(dynamics.sys)))
        p_syms = Symbol.(parameters(dynamics.sys))
        (s_syms, p_syms)
    elseif dynamics.sys isa SciMLBase.SymbolCache
        s_syms = SciMLBase.variable_symbols(dynamics.sys)
        p_syms = if isnothing(dynamics.sys.parameters)
            []
        else
            dynamics.sys.parameters
        end
        (s_syms, p_syms)
    else
        ([], [])
    end

    return NeuralLyapunovPDESystem(
        dynamics.f,
        lb,
        ub,
        spec;
        fixed_point = fixed_point,
        p = p,
        state_syms = s_syms,
        parameter_syms = p_syms,
        policy_search = false
    )
end

function NeuralLyapunovPDESystem(
        dynamics::ODESystem,
        bounds,
        spec::NeuralLyapunovSpecification;
        fixed_point = zeros(length(bounds))
)::Tuple{PDESystem, Function}
    ######################### Check for policy search #########################
    f, x, p, policy_search = if isempty(ModelingToolkit.unbound_inputs(dynamics))
        (ODEFunction(dynamics), states(dynamics), parameters(dynamics), false)
    else
        (f, _), x, p = ModelingToolkit.generate_control_function(dynamics; simplify = true)
        (f, x, p, true)
    end

    ########################## Define state symbols ###########################
    # States should all be functions of time, but we just want the symbol
    # e.g., if the state is ω(t), we just want ω
    _state = operation.(x)
    state_syms = Symbol.(_state)
    state = [first(@parameters $s) for s in state_syms]

    ###################### Remove derivatives in domains ######################
    domains = map(
        d -> Num(operation(Symbolics.diff2term(Symbolics.value(d.variables)))) ∈ d.domain,
        bounds
    )
    domain_vars = map(d -> d.variables, domains)
    if Set(_state) != Set(domain_vars)
        error("Domain variables from `domains` do not match those extracted from " *
              "`dynamics`. Got $_state from `dynamics` and $domain_vars from `domains`.")
    end

    ########################### Construct PDESystem ###########################
    _NeuralLyapunovPDESystem(
        f,
        domains,
        spec,
        fixed_point,
        state,
        p,
        ModelingToolkit.get_defaults(dynamics),
        policy_search
    )
end

function _NeuralLyapunovPDESystem(
        dynamics::Function,
        domains,
        spec::NeuralLyapunovSpecification,
        fixed_point,
        state,
        params,
        defaults,
        policy_search::Bool
)::Tuple{PDESystem, Function}
    ########################## Unpack specifications ##########################
    structure = spec.structure
    minimzation_condition = spec.minimzation_condition
    decrease_condition = spec.decrease_condition
    f_call = structure.f_call
    state_dim = length(domains)

    ################## Define Lyapunov function & derivative ##################
    output_dim = structure.network_dim
    net_syms = [Symbol(:φ, i) for i in 1:output_dim]
    net = [first(@variables $s(..)) for s in net_syms]

    # φ(x) is the symbolic form of neural network output
    φ(x) = Num.([φi(x...) for φi in net])

    # V_sym(x) is the symobolic form of the Lyapunov function
    V_sym(x) = structure.V(φ, x, fixed_point)

    # V̇_sym(x) is the symbolic time derivative of the Lyapunov function
    function V̇_sym(x)
        structure.V̇(
            φ,
            y -> Symbolics.jacobian(φ(y), y),
            dynamics,
            x,
            params,
            0.0,
            fixed_point
        )
    end

    ################ Define equations and boundary conditions #################
    eqs = []

    if check_nonnegativity(minimzation_condition)
        cond = get_minimization_condition(minimzation_condition)
        push!(eqs, cond(V_sym, state, fixed_point) ~ 0.0)
    end

    if check_decrease(decrease_condition)
        cond = get_decrease_condition(decrease_condition)
        push!(eqs, cond(V_sym, V̇_sym, state, fixed_point) ~ 0.0)
    end

    bcs = []

    if check_minimal_fixed_point(minimzation_condition)
        push!(bcs, V_sym(fixed_point) ~ 0.0)
    end

    if policy_search
        append!(bcs, f_call(dynamics, φ, fixed_point, params, 0.0) .~ zeros(state_dim))
    end

    if isempty(eqs) && isempty(bcs)
        error("No training conditions specified.")
    end

    # NeuralPDE requires an equation and a boundary condition, even if they are
    # trivial like 0.0 == 0.0, so we remove those trivial equations if they showed up
    # naturally alongside other equations and add them in if we have no other equations
    eqs = filter(eq -> eq != (0.0 ~ 0.0), eqs)
    bcs = filter(eq -> eq != (0.0 ~ 0.0), bcs)

    if isempty(eqs)
        push!(eqs, 0.0 ~ 0.0)
    end
    if isempty(bcs)
        push!(bcs, 0.0 ~ 0.0)
    end

    ########################### Construct PDESystem ###########################
    @named lyapunov_pde_system = PDESystem(
        eqs,
        bcs,
        domains,
        state,
        φ(state),
        params;
        defaults = defaults
    )

    ################### Return PDESystem and neural network ###################
    # φ_func is the numerical form of neural network output
    function φ_func(phi, θ, x)
        reduce(
            vcat,
            Array(phi[i](x, θ.depvar[net_syms[i]])) for i in 1:output_dim
        )
    end

    return lyapunov_pde_system, φ_func
end

"""
    NumericalNeuralLyapunovFunctions(phi, θ, network_func, structure, dynamics, fixed_point;
                                     jac, J_net)

Returns the Lyapunov function, its time derivative, and its gradient: `V(state)`,
`V̇(state)`, and `∇V(state)`

These functions can operate on a state vector or columnwise on a matrix of state vectors.
`phi` is the neural network with parameters `θ`. `network_func(phi, θ, state)` is an output
of `NeuralLyapunovPDESystem`, which evaluates the neural network represented by `phi` with
parameters `θ` at `state`.

The Lyapunov function structure is specified in structure, which is a
`NeuralLyapunovStructure`. The Jacobian of the network is either specified via
`J_net(_phi, _θ, state)` or calculated using `jac`, which defaults to
`ForwardDiff.jacobian`.
"""
function NumericalNeuralLyapunovFunctions(
        phi,
        θ,
        network_func::Function,
        structure::NeuralLyapunovStructure,
        dynamics::Function,
        fixed_point;
        p = SciMLBase.NullParameters(),
        jac = ForwardDiff.jacobian,
        J_net = (_phi, _θ, x) -> jac((y) -> network_func(_phi, _θ, y), x)
)::Tuple{Function, Function, Function}
    # Make Network function
    _net_func = (x) -> network_func(phi, θ, x)
    _J_net = (x) -> J_net(phi, θ, x)

    # Numerical form of Lyapunov function
    V_func(state::AbstractVector) = structure.V(_net_func, state, fixed_point)
    V_func(state::AbstractMatrix) = mapslices(V_func, state, dims = [1])

    # Numerical gradient of Lyapunov function
    ∇V_func(state::AbstractVector) = structure.∇V(
        _net_func,
        _J_net,
        state,
        fixed_point
    )
    ∇V_func(state::AbstractMatrix) = mapslices(∇V_func, state, dims = [1])

    # Numerical time derivative of Lyapunov function
    function V̇_func(state::AbstractVector)
        structure.V̇(
            _net_func,
            _J_net,
            dynamics,
            state,
            p,
            0.0,
            fixed_point
        )
    end
    V̇_func(state::AbstractMatrix) = mapslices(V̇_func, state, dims = [1])

    return V_func, V̇_func, ∇V_func
end

"""
    NumericalNeuralLyapunovFunctions(phi, θ, network_func, V_structure, dynamics,
                                     fixed_point, grad)

Returns the Lyapunov function, its time derivative, and its gradient: `V(state)`,
`V̇(state)`, and `∇V(state)`.

These functions can operate on a state vector or columnwise on a matrix of state vectors.
`phi` is the neural network with parameters `θ`. `network_func` is an output of
`NeuralLyapunovPDESystem`.

The Lyapunov function structure is defined by
    `V_structure(_network_func, state, fixed_point)`
Its gradient is calculated using `grad`, which defaults to `ForwardDiff.gradient`.
"""
function NumericalNeuralLyapunovFunctions(
        phi,
        θ,
        network_func::Function,
        V_structure::Function,
        dynamics::Function,
        fixed_point;
        p = SciMLBase.NullParameters(),
        grad = ForwardDiff.gradient
)::Tuple{Function, Function, Function}
    # Make network function
    _net_func = (x) -> network_func(phi, θ, x)

    # Numerical form of Lyapunov function
    V_func(state::AbstractVector) = V_structure(_net_func, state, fixed_point)
    V_func(state::AbstractMatrix) = mapslices(V_func, state, dims = [1])

    # Numerical gradient of Lyapunov function
    ∇V_func(state::AbstractVector) = grad(V_func, state)
    ∇V_func(state::AbstractMatrix) = mapslices(∇V_func, state, dims = [1])

    # Numerical time derivative of Lyapunov function
    V̇_func(state::AbstractVector) = dynamics(state, p, 0.0) ⋅ ∇V_func(state)
    V̇_func(state::AbstractMatrix) = mapslices(V̇_func, state, dims = [1])

    return V_func, V̇_func, ∇V_func
end
