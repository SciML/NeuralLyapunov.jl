"""
    NeuralLyapunovPDESystem(dynamics, lb, ub, spec; fixed_point)

Constructs a ModelingToolkit PDESystem to train a neural Lyapunov function

Returns the PDESystem and a function representing the neural network, which
operates columnwise.

The neural Lyapunov function will only be trained for { x : lb .≤ x .≤ ub }.
The Lyapunov function will be for the dynamical system represented by dynamics
If dynamics is an ODEProblem, then the corresponding ODE; if dynamics is a 
function, then the ODE is ẋ = dynamics(x). This ODE should have a fixed point
at x = fixed_point. The particular Lyapunov conditions to be used and structure
of the neural Lyapunov function are specified through spec, which is a 
NeuralLyapunovSpecification.

The returned neural network function takes three inputs: the neural network 
structure phi, the trained parameters res, and a matrix of inputs to operate on
columnwise.
"""
function NeuralLyapunovPDESystem(
    dynamics::Function,
    lb,
    ub,
    spec::NeuralLyapunovSpecification;
    fixed_point = zeros(length(lb)),
)::Tuple{PDESystem,Function}
    ########################## Unpack specifications ##########################
    structure = spec.structure
    minimzation_condition = spec.minimzation_condition
    decrease_condition = spec.decrease_condition

    ######################### Define state symbols ############################
    state_dim = length(lb)
    state_syms = [Symbol(:state, i) for i = 1:state_dim]

    # Create a vector of ModelingToolkit parameters representing the state
    state = [first(@parameters $s) for s in state_syms]

    ############################# Define domains ##############################
    domains = [state[i] ∈ (lb[i], ub[i]) for i = 1:state_dim]

    ################## Define Lyapunov function & derivative ##################
    output_dim = structure.network_dim
    net_syms = [Symbol(:u, i) for i = 1:output_dim]
    net = [first(@variables $s(..)) for s in net_syms]

    # u(x) is the symbolic form of neural network output
    u(x) = Num.([ui(x...) for ui in net])

    # V_sym(x) is the symobolic form of the Lyapunov function
    V_sym(x) = structure.V(u, x, fixed_point)

    # V̇_sym(x) is the symbolic time derivative of the Lyapunov function
    V̇_sym(x) = structure.V̇(u, y -> Symbolics.jacobian(u(y), y), dynamics, x, fixed_point)

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
    
    if check_fixed_point(minimzation_condition)
        push!(bcs, V_sym(fixed_point) ~ 0.0)
    end
    if check_stationary_fixed_point(decrease_condition)
        push!(bcs, V̇_sym(fixed_point) ~ 0.0)
    end

    if isempty(eqs) && isempty(bcs)
        error("No training conditions specified.")
    end

    ########################### Construct PDESystem ###########################
    @named lyapunov_pde_system = PDESystem(eqs, bcs, domains, state, u(state))

    ################### Return PDESystem and neural network ###################
    # u_func is the numerical form of neural network output
    u_func(phi, res, x) = reduce(
        vcat, 
        Array(phi[i](x, res.u.depvar[net_syms[i]])) for i = 1:output_dim
        )

    return lyapunov_pde_system, u_func
end

function NeuralLyapunovPDESystem(
    dynamics::ODEProblem,
    lb,
    ub,
    spec::NeuralLyapunovSpecification;
    fixed_point = zeros(length(lb)),
)::Tuple{PDESystem,Function}
    f = get_dynamics_from_ODEProblem(dynamics)
    return NeuralLyapunovPDESystem(f, lb, ub, spec; fixed_point)
end

"""
    NumericalNeuralLyapunovFunctions(phi, result, network_func, structure, dynamics, fixed_point; jac, J_net)

Returns the Lyapunov function, its time derivative, and its gradient: V(state), 
V̇(state), and ∇V(state)

These functions can operate on a state vector or columnwise on a matrix of state
vectors. phi is the neural network with parameters in result. 
network_func(phi, res, state) is an output of NeuralLyapunovPDESystem, which 
evaluates the neural network represented phi with parameters res at state.

The Lyapunov function structure is specified in structure, which is a 
NeuralLyapunovStructure. The Jacobian of the network is either specified via
J_net(_phi, _result, state) or calculated using jac, which defaults to 
ForwardDiff.jacobian
"""
function NumericalNeuralLyapunovFunctions(
    phi,
    result,
    network_func::Function,
    structure::NeuralLyapunovStructure,
    dynamics::Function,
    fixed_point;
    jac = ForwardDiff.jacobian,
    J_net = (_phi, _res, x) -> jac((y) -> network_func(_phi, _res, y), x)
)::Tuple{Function, Function, Function}
    # Make Network function
    _net_func = (x) -> network_func(phi, result, x)
    _J_net = (x) -> J_net(phi, result, x)

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
    V̇_func(state::AbstractVector) = structure.V̇(
        _net_func, 
        _J_net, 
        dynamics, 
        state, 
        fixed_point
        )
    V̇_func(state::AbstractMatrix) = mapslices(V̇_func, state, dims = [1])

    return V_func, V̇_func, ∇V_func
end

"""
    NumericalNeuralLyapunovFunctions(phi, result, network_func, V_structure, dynamics, fixed_point, grad)

Returns the Lyapunov function, its time derivative, and its gradient: V(state), 
V̇(state), and ∇V(state)

These functions can operate on a state vector or columnwise on a matrix of state
vectors. phi is the neural network with parameters in result. network_func is 
an output of NeuralLyapunovPDESystem.

The Lyapunov function structure is defined by 
    V_structure(_network_func, state, fixed_point)
Its gradient is calculated using grad, which defaults to ForwardDiff.gradient. 
"""
function NumericalNeuralLyapunovFunctions(
    phi,
    result,
    network_func,
    V_structure::Function,
    dynamics::Function,
    fixed_point,
    grad = ForwardDiff.gradient,
)::Tuple{Function, Function, Function}
    # Make network function
    _net_func = (x) -> network_func(phi, result, x)

    # Numerical form of Lyapunov function
    V_func(state::AbstractVector) = V_structure(_net_func, state, fixed_point)
    V_func(state::AbstractMatrix) = mapslices(V_func, state, dims = [1])

    # Numerical gradient of Lyapunov function
    ∇V_func(state::AbstractVector) = grad(V_func, state)
    ∇V_func(state::AbstractMatrix) = mapslices(∇V_func, state, dims = [1])

    # Numerical time derivative of Lyapunov function
    V̇_func(state::AbstractVector) = dynamics(state) ⋅ ∇V_func(state)
    V̇_func(state::AbstractMatrix) = mapslices(V̇_func, state, dims = [1])
    #= # This version might actually be slower; unsure
    V̇_func(state::AbstractMatrix) = reshape(
        map(
            x -> x[1] ⋅ x[2],
            zip(eachslice(dynamics(state), dims = 2), eachslice(∇V_func(state), dims = 2)),
        ),
        (1, :),
    )
    =#

    return V_func, V̇_func, ∇V_func
end

#=
function NumericalNeuralLyapunovFunctions(
    phi,
    result,
    lyapunov_func,
    dynamics::ODEProblem;
    grad = ForwardDiff.gradient,
)::Tuple{Function, Function, Function}
    f = get_dynamics_from_ODEProblem(dynamics)
    return NumericalNeuralLyapunovFunctions(phi, result, lyapunov_func, f; grad)
end
=#


"""
    get_dynamics_from_ODEProblem(prob)
Extracts f such that ODEProblem is ẋ = f(x)

The returned function f can operate on a single x vector or columnwise on a 
matrix of x values.
"""
function get_dynamics_from_ODEProblem(prob::ODEProblem)::Function
    dynamicsODEfunc = prob.f
    f_ = if dynamicsODEfunc.mass_matrix == I
        state -> dynamicsODEfunc.f(state, prob.p, 0.0) # Let time be 0.0, since we're only considering time-invariant dynamics
    else
        throw(ErrorException("DAEs are not supported at this time"))
    end
    f(state::AbstractVector) = f_(state)
    f(state::AbstractMatrix) = mapslices(f_, state, dims = [1])
    return f
end
