"""
    get_quadratic_lyapunov_function(dynamics; <keyword_arguments>)

Get a quadratic Lyapunov function for the linearization of the given system around the fixed
point. Return the Lyapunov function and its time derivative.

When the system is closed-loop, the continuous-time Lyapunov equation
(``A^T P + P A + Q = 0``) is solved directly using the supplied `Q` (defaults to the identity
matrix). The resulting Lyapunov function is then ``V(x) = (x - x_0)^T P (x - x_0)``.

When the system is open-loop, the LQR problem is solved to find a linear feedback controller
and its associated quadratic Lyapunov function.

# Positional Arguments
  - `dynamics`: the dynamical system being analyzed, represented as a `System` or the
    function `f` such that `ẋ = f(x[, u], p, t)`; either way, the ODE should not depend on
    time and only `t = 0` will be used. If `dynamics isa System`, call
    `mtkcompile(dynamics)` before `NeuralLyapunovPDESystem`, or
    `mtkcompile(dynamics; inputs = ..., split = false)` if the system has unbound inputs.

# Keyword Arguments
  - `fixed_point`: the equilibrium being analyzed, around which to linearize the system
    (defaults to the origin, dimension inferred from `Q` or `x_dim` if available).
  - `Q`: The weighting matrix for the state in the Lyapunov equation or the LQR problem
    (defaults to the identity matrix).
  - `R`: The weighting matrix for the control input in the LQR problem (only used when the
    system is open-loop; defaults to the identity matrix).
  - `p`: the values of the parameters of the dynamical system being analyzed; defaults to
    `SciMLBase.NullParameters()`; not used when `dynamics isa System`, then use the
    default parameter values of `dynamics`.
  - `u_eq`: The equilibrium control input (only used when the system is open-loop`; defaults
    to zero control input).
  - `u_dim`: The dimension of the control vector (only used when the system is open-loop and
    `!(dynamics isa System)`); inferred from `R` or `u_eq` if available.
  - `x_dim`: The dimension of the state vector (only used when `dynamics` is not a `System`
    nor an `ODEFunction` nor an `ODEInputFunction`); inferred from `Q` or `fixed_point` if
    available.
"""
function get_quadratic_lyapunov_function(
    dynamics::System;
    fixed_point = zeros(length(unknowns(dynamics))),
    u_eq = zeros(length(unbound_inputs(dynamics))),
    Q = I(length(unknowns(dynamics))),
    R = I(length(unbound_inputs(dynamics)))
)
    # Check if system is closed-loop
    closed_loop = isempty(unbound_inputs(dynamics))

    # Extract parameter values
    params = setdiff(parameters(dynamics), unbound_inputs(dynamics))
    p = [Symbolics.value(initial_conditions(dynamics)[p]) for p in params]
    T = eltype(fixed_point)
    t0 = zero(T)

    # Check for consistent Q and fixed_point size
    n = checksquare(Q)
    x_dim = length(fixed_point)
    _x_dim = length(unknowns(dynamics))
    if !(n == x_dim == _x_dim)
        throw(
            ArgumentError(
                "Inconsistent dimensions between Q ($n, $n), fixed_point ($x_dim), and " *
                "unknowns of dynamics ($_x_dim)."
            )
        )
    end

    if closed_loop
        # Solve Lyapunov equation
        f = ODEFunction(dynamics; jac = true)
        A = f.jac(fixed_point, p, t0)
        P = lyapc(A', Q)
    else
        # Check for consistent R and u_dim size
        u_dim = length(unbound_inputs(dynamics))
        n = checksquare(R)
        if n != u_dim
            throw(
                ArgumentError(
                    "Inconsistent dimensions between R ($n, $n) and unbound inputs of " *
                    "dynamics ($u_dim)."
                )
            )
        end

        # Solve the LQR problem via Riccati equation
        #= I think this should work, but is currently erroring with jac = true and/or
        # controljac = true, so we just use the ModelingToolkit internals to generate the
        # Jacobians
        f = ODEInputFunction(dynamics; jac = true, controljac = true)
        A = f.jac(fixed_point, u_eq, p, t0)
        B = f.controljac(fixed_point, u_eq, p, t0)
        =#
        J, = generate_jacobian(dynamics; expression = Val{false})
        CJ, = generate_control_jacobian(dynamics; expression = Val{false})
        A = J(fixed_point, vcat(p, u_eq), t0)
        B = CJ(fixed_point, vcat(p, u_eq), t0)
        P, _, K, _, _ = arec(A, B, R, Q)

        # Generate closed-loop dynamics
        f = let _K = copy(K), x0 = copy(fixed_point), u0 = copy(u_eq),
            _f = ODEInputFunction(dynamics)
            (x, _p, t) -> _f(x, -_K * (x - x0) + u0, _p, t)
        end
    end

    return numerical_local_lyapunov_functions(f, fixed_point, P, p, t0)
end

function get_quadratic_lyapunov_function(
    dynamics::ODEFunction;
    fixed_point = nothing,
    p = SciMLBase.NullParameters(),
    Q = nothing
)
    # Determine the fixed point if it's not provided
    if fixed_point === nothing
        if dynamics.sys isa System
            fixed_point = zeros(length(unknowns(dynamics)))
        elseif dynamics.sys isa SymbolCache
            fixed_point = zeros(length(variable_symbols(dynamics.sys)))
        else
            throw(ArgumentError("Cannot determine fixed point for the given dynamics."))
        end
    end

    t0 = zero(eltype(fixed_point))

    # Determine the Q matrix if it's not provided
    if Q === nothing
        Q = I(length(fixed_point))
    end

    # Check for consistent Q and fixed_point size
    n = checksquare(Q)
    x_dim = length(fixed_point)
    if n != x_dim
        throw(
            ArgumentError(
                "Inconsistent dimensions between Q ($n, $n) and fixed_point ($x_dim).)"
            )
        )
    end

    # Linearize the system
    if SciMLBase.__has_jac(dynamics) && dynamics.jac !== nothing
        A = dynamics.jac(fixed_point, p, t0)
    else
        A = ForwardDiff.jacobian(x -> dynamics(x, p, t0), fixed_point)
    end

    # Solve the Lyapunov equation
    P = lyapc(A', Q)

    # Return the Lyapunov function and its time derivative
    return numerical_local_lyapunov_functions(dynamics, fixed_point, P, p, t0)
end

function get_quadratic_lyapunov_function(
    dynamics::ODEInputFunction;
    fixed_point = nothing,
    u_eq = nothing,
    p = SciMLBase.NullParameters(),
    Q = nothing,
    R = nothing,
    u_dim = 0
)
    # Determine the fixed point if it's not provided
    if fixed_point === nothing
        if dynamics.sys isa System
            fixed_point = zeros(length(unknowns(dynamics.sys)))
        elseif dynamics.sys isa SymbolCache
            fixed_point = zeros(length(variable_symbols(dynamics.sys)))
        else
            throw(ArgumentError("Cannot determine fixed point for the given dynamics."))
        end
    end
    x_dim = length(fixed_point)
    T = eltype(fixed_point)
    t0 = zero(T)

    # Determine the Q matrix if it's not provided
    if Q === nothing
        Q = I(length(fixed_point))
    end

    # Check for consistent Q and fixed_point size
    n = checksquare(Q)
    if n != x_dim
        throw(
            ArgumentError(
                "Inconsistent dimensions between Q ($n, $n) and fixed_point ($x_dim)."
            )
        )
    end

    # Infer and check consistency of u_dim and R
    if dynamics.sys isa System
        _u_dim = length(unbound_inputs(dynamics.sys))
        if u_dim > 0 && u_dim != _u_dim
            throw(
                ArgumentError(
                    "Inconsistent dimensions between u_dim ($u_dim) and " *
                    "unbound inputs of underlying System ($_u_dim)."
                )
            )
        end
        u_dim = _u_dim

        if u_eq !== nothing
            n = length(u_eq)
            if n != u_dim
                throw(
                    ArgumentError(
                        "Inconsistent dimensions between u_dim ($u_dim) and u_eq " *
                        "($(length(u_eq)))."
                    )
                )
            end
        else
            u_eq = zeros(T, u_dim)
        end

        if R !== nothing
            n = checksquare(R)
            if n != u_dim
                throw(
                    ArgumentError(
                        "Inconsistent dimensions between R ($n, $n) and unbound " *
                        "inputs of underlying System ($u_dim)."
                    )
                )
            end
        else
            R = I(u_dim)
        end
    else
        if u_dim ≤ 0 && R === nothing && u_eq === nothing
            throw(
                ArgumentError(
                    "u_dim could not be inferred from the provided arguments. Please " *
                    "provide u_eq, u_dim, or a valid R matrix."
                )
            )
        end

        if u_eq === nothing && R === nothing
            u_eq = zeros(T, u_dim)
            R = I(u_dim)
        elseif u_eq !== nothing
            if u_dim ≤ 0
                u_dim = length(u_eq)
            elseif u_dim != length(u_eq)
                throw(
                    ArgumentError(
                        "Inconsistent dimensions between u_dim ($u_dim) and u_eq " *
                        "($(length(u_eq)))."
                    )
                )
            end

            if R === nothing
                R = I(u_dim)
            else
                n = checksquare(R)
                if n != u_dim
                    throw(
                        ArgumentError(
                            "Inconsistent dimensions between R ($n, $n) and u_eq ($u_dim)."
                        )
                    )
                end
            end
        else # u_eq === nothing && R !== nothing
            n = checksquare(R)

            if u_dim ≤ 0
                u_dim = n
            elseif u_dim != n
                throw(
                    ArgumentError(
                        "Inconsistent dimensions between u_dim ($u_dim) and R ($n, $n)."
                    )
                )
            end

            u_eq = zeros(T, u_dim)
        end
    end

    # Linearize dynamics
    if SciMLBase.__has_jac(dynamics) && dynamics.jac !== nothing
        A = dynamics.jac(fixed_point, u_eq, p, t0)
    else
        A = ForwardDiff.jacobian(x -> dynamics(x, u_eq, p, t0), fixed_point)
    end

    if SciMLBase.__has_controljac(dynamics) && dynamics.controljac !== nothing
        B = dynamics.controljac(fixed_point, u_eq, p, t0)
    else
        B = ForwardDiff.jacobian(u -> dynamics(fixed_point, u, p, t0), u_eq)
    end

    # Solve the LQR problem via Riccati equation
    P, _, K, _, _ = arec(A, B, R, Q)

    # Generate closed-loop dynamics
    f = let _K = copy(K), x0 = copy(fixed_point), u0 = copy(u_eq)
        (x, p, t) -> dynamics(x, u0 - _K * (x - x0), p, t)
    end

    return numerical_local_lyapunov_functions(f, fixed_point, P, p, t0)
end

function get_quadratic_lyapunov_function(
    dynamics;
    fixed_point = nothing,
    u_eq = nothing,
    p = SciMLBase.NullParameters(),
    Q = nothing,
    R = nothing,
    x_dim = 0,
    u_dim = 0
)
    # Infer and check consistency of fixed_point, Q, and x_dim
    if fixed_point === nothing
        if Q !== nothing
            n = checksquare(Q)
            if x_dim > 0 && n != x_dim
                throw(
                    ArgumentError(
                        "Inconsistent dimensions between Q ($n, $n) and x_dim ($x_dim)."
                    )
                )
            else
                x_dim = n
            end
        elseif x_dim ≤ 0
            throw(
                ArgumentError(
                    "x_dim could not be inferred from the provided arguments. Please " *
                    "provide fixed_point, Q, or a positive x_dim."
                )
            )
        end

        fixed_point = zeros(x_dim)
        Q = I(x_dim)
    else
        if x_dim ≤ 0
            x_dim = length(fixed_point)
        elseif x_dim != length(fixed_point)
            throw(
                ArgumentError(
                    "Inconsistent dimensions between x_dim ($x_dim) and fixed_point " *
                    "($(length(fixed_point)))."
                )
            )
        end

        if Q !== nothing
            n = checksquare(Q)
            if n != x_dim
                throw(
                    ArgumentError(
                        "Inconsistent dimensions between Q ($n, $m) and fixed_point ($x_dim)."
                    )
                )
            end
        else
            Q = I(x_dim)
        end

    end

    T = eltype(fixed_point)

    # Infer and check consistency of u_dim and R
    lqr = u_dim > 0 || R !== nothing || u_eq !== nothing
    if lqr
        if u_eq === nothing && R === nothing
            u_eq = zeros(T, u_dim)
            R = I(u_dim)
        elseif u_eq !== nothing
            if u_dim ≤ 0
                u_dim = length(u_eq)
            elseif u_dim != length(u_eq)
                throw(
                    ArgumentError(
                        "Inconsistent dimensions between u_dim ($u_dim) and u_eq " *
                        "($(length(u_eq)))."
                    )
                )
            end

            if R === nothing
                R = I(u_dim)
            else
                n = checksquare(R)
                if n != u_dim
                    throw(
                        ArgumentError(
                            "Inconsistent dimensions between R ($n, $n) and u_eq ($u_dim)."
                        )
                    )
                end
            end
        else # u_eq === nothing && R !== nothing
            n = checksquare(R)

            if u_dim ≤ 0
                u_dim = n
            elseif u_dim != n
                throw(
                    ArgumentError(
                        "Inconsistent dimensions between u_dim ($u_dim) and R ($n, $n)."
                    )
                )
            end

            u_eq = zeros(T, u_dim)
        end
    end

    t0 = zero(T)
    if lqr
        # Linearize dynamics
        A = ForwardDiff.jacobian(x -> dynamics(x, u_eq, p, t0), fixed_point)
        B = ForwardDiff.jacobian(u -> dynamics(fixed_point, u, p, t0), u_eq)

        # Solve the LQR problem via Riccati equation
        P, _, K, _, _ = arec(A, B, R, Q)

        P

        # Generate closed-loop dynamics
        f = let _K = copy(K), x0 = copy(fixed_point), u0 = copy(u_eq)
            (x, p, t) -> dynamics(x, u0 -_K * (x - x0), p, t)
        end
    else
        # Linearize dynamics
        A = ForwardDiff.jacobian(x -> dynamics(x, p, t0), fixed_point)

        # Solve the Lyapunov equation
        P = lyapc(A', Q)

        f = dynamics
    end

    return numerical_local_lyapunov_functions(f, fixed_point, P, p, t0)
end

function numerical_local_lyapunov_functions(f, fixed_point, P, p, t0)
    # Return the Lyapunov function and its time derivative
    let x0 = copy(fixed_point), _P = copy(P), _p = copy(p)
        # Numerical form of Lyapunov function
        V(x::AbstractVector) = dot(x - x0, _P, x - x0)
        V(x::AbstractMatrix) = mapslices(V, x, dims = [1])

        # Numerical gradient of Lyapunov function
        #    ∇V(x::AbstractVector) = 2 * (_P * (x - x0))
        #    ∇V(x::AbstractMatrix) = mapslices(∇V, x, dims = [1])

        # Numerical time derivative of Lyapunov function
        V̇(x::AbstractVector) = 2 * dot(f(x, _p, t0), _P, x - x0)
        V̇(x::AbstractMatrix) = mapslices(V̇, x, dims = [1])

        return V, V̇
    end
end
