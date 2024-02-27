"""
    get_RoA_estimate(V, dVdt, lb, ub; fixed_point, ∇V)

Finds the level of the largest sublevelset comletely in the domain in which the
Lyapunov conditions are met. Specifically finds the largest `ρ` such that
    `V(x) < ρ => lb .< x .< ub && (dVdt(x) < 0 || x = fixed_point)`

The fixed point defaults to the origin, with dimension inferred from `lb`. If `∇V`
is not specified, it is calculated using `ForwardDiff`.
"""
function get_RoA_estimate(V, dVdt, lb, ub; reltol = 1e-3, abstol = 1e-4,
                          fixed_point = zeros(length(lb)), ∇V = nothing)
    state_dim = length(lb)
    if isnothing(∇V)
        ∇V_func(state::AbstractVector) = ForwardDiff.gradient(V, state)
        ∇V_func(states::AbstractMatrix) = mapslices(∇V_func, states, dims = [1])
    else
        ∇V_func = ∇V
    end
    _get_RoA_estimate(V, dVdt, state_dim, lb, ub, fixed_point, ∇V_func, reltol, abstol)
end

function _get_RoA_estimate(V, dVdt, state_dim, lb, ub, fixed_point, ∇V, reltol, abstol)
    # Let ρ_max = minimum value of V on the boundary
    ρ_max, x_opt, i_bd = find_min_on_bounding_box(V, state_dim, lb, ub, ∇V)

    # Move x_opt just off the boundary
    x_opt[i_bd] -= √eps(typeof(x_opt[i_bd])) * sign(x_opt[i_bd] - fixed_point[i_bd])

    # Initialzie search for max ρ : ( (max V̇(x) : V(x) < ρ) < 0)
    # We cannot verify V(x) > ρ_max, since some such x are outside the domain
    # We have verified V(x) ≤ 0, since the only such x is fixed_point
    # So, our search is for ρ ∈ [ρ_min, ρ_max] and we begin by attempting to
    # validate the whole set {x : V(x) < ρ_max}
    ρ_min = 0.0
    ρ = ρ_max

    # Set up max V̇(x) : V(x) < ρ OptimizationFunction
    negV̇_param = make_paramerized_scalar_func(x -> -dVdt(x))
    V_param = make_paramerized_scalar_func(V)
    ∇V_param = make_paramerized_vector_func(∇V)

    f = OptimizationFunction{true}(
        negV̇_param,
        Optimization.AutoFiniteDiff();
        cons = V_param,
        cons_j = ∇V_param,
    )

    # Stop the search early if we find any x : V̇(x) > 0
    cb = (x, negdVdt) -> negdVdt < 0 # i.e., V̇ > 0

    # Binary search
    while abs(ρ_max - ρ_min) > max(abstol, reltol*ρ_max)
        # Find a point just interior of the boundary to start optimization
        guess = initialize_guess(V_param, lb, ub, ρ_min, ρ, x_opt; ∇V = ∇V_param)

        # Find max V̇(x) : ρ_min ≤ V(x) < ρ
        # We've already verified V(x) < ρ_min, so don't need to check there
        prob = OptimizationProblem{true}(
            f,
            guess,
            lb = lb,
            ub = ub,
            lcons = [ρ_min],
            ucons = [ρ],
            callback = cb,
        )
        opt = OptimizationOptimJL.IPNewton()
        res = solve(prob, opt, allow_f_increases = true, successive_f_tol = 2)
        V̇_max = dVdt(res.u)

        if V̇_max > √eps(Float64)
            # Since V̇(x_opt) > 0, we cannot verify any x : V(x) ≥ V(x_opt),
            # so we limit our search to ρ ∈ [ρ_min, V(x_opt)]
            x_opt = res.u
            ρ_max = V(x_opt)
        else
            # Since (max V̇(x) : V(x) < ρ) < 0, we have verified {x : V(x) < ρ}
            # We continue our search of [ρ, ρ_max] to see if we can do better
            ρ_min = ρ
        end
        ρ = (ρ_max + ρ_min) / 2
    end
    return ρ_min
end

"""
    initialize_guess(V, lb, ub, ρ_min, ρ, x_opt; ∇V = nothing)
Finds a point `x` of the same shape as `x_opt` such that `ρ_min < V(x) < ρ`.
"""
function initialize_guess(V, lb, ub, ρ_min, ρ, x_opt; ∇V = nothing)
    f = OptimizationFunction{true}(
        V,
        Optimization.AutoFiniteDiff();
        grad = ∇V,
        cons = V,
        cons_j = ∇V,
    )
    prob = OptimizationProblem{true}(
        f,
        x_opt,
        lb = lb,
        ub = ub,
        lcons = [ρ_min],
        ucons = [Inf]
    )
    opt = OptimizationOptimJL.IPNewton()
    cb = (x, V) -> V < ρ
    res = solve(prob, opt, allow_f_increases = true, successive_f_tol = 2, callback=cb)
    return res.u
end

"""
    initialize_guess_aided(V, cons, boundary_val, lb, ub, ρ_min, ρ, x_start, ∇V, ∇cons)

Finds a point x of the same shape as x_start such that `ρ_min < V(x) < ρ` and
`cons(x)[2] > boundary_val`.

Assumes `cons(⋅)[1] == V(⋅), V(x_start) > ρ, cons(x_start)[2] > boundary_val`.
Also, `V` should actually be `V(x,p)` and `cons` should be `cons(x,p)` where `p` are
unused parameters.
"""
function initialize_guess_aided(V, cons, boundary_val, lb, ub, ρ_min, ρ, x_start, ∇V,
                                cons_j)
    if ρ_min < V(x_start, nothing) < ρ && cons(x_start, nothing)[2] > boundary_val
        return x_start
    end
    if V(x_start, nothing) < ρ_min
        @show x_start, V(x_start, nothing), ρ_min
        throw("Invalid starting guess, V < ρ_min")
    end
    if any(x_start .< lb) || any(x_start .> ub)
        @show lb, x_start, ub
        throw("Invalid starting guess, outside box")
    end
    if cons(x_start, nothing)[2] < boundary_val
        @show x_start, cons(x_start, nothing)[2], boundary_val
        throw("Invalid starting guess, constraint not met")
    end
    f = OptimizationFunction{true}(
        V,
        Optimization.AutoFiniteDiff();
        grad = ∇V,
        cons = cons,
        cons_j = cons_j,
    )
    prob = OptimizationProblem{true}(
        f,
        x_start,
        lb = lb,
        ub = ub,
        lcons = [ρ_min, boundary_val],
        ucons = [Inf, Inf]
    )
    opt = OptimizationOptimJL.IPNewton(μ0=0.1)
    cb = (x, V) -> ρ_min < V < ρ
    res = solve(prob, opt, allow_f_increases = true, successive_f_tol = 2, callback=cb)
    if ρ_min < V(res.u, nothing) < ρ && cons(res.u, nothing)[2] > boundary_val
        return res.u
    elseif cons(res.u, nothing)[2] > boundary_val
        @show res.u, ρ_min, V(res.u, nothing), ρ
        throw("Generated bad initial guess, V ∉ (ρ_min, ρ)")
    else
        @show res.u, cons(res.u, nothing)[2], boundary_val
        throw("Generated bad initial guess, picked point in verified region")
    end
end

"""
    find_min_on_boundary(V, state_dim, lb, ub, ∇V)

Finds the minimum value of `V` on the bounding box given by
    `any(x .== lb) || any(x .== ub)`

Returns `(V_min, x_min, i_bd)` such that `V(x_min) = V_min` is the minimum value of
`V` on the boundary and `x_min[i_bd] == lb[i_bd] || x_min[i_bd] == ub[i_bd]`.
"""
function find_min_on_bounding_box(V, state_dim, lb, ub, ∇V)
    # TODO: @view to speed up?
    candidates = Vector{Any}(undef, 2 * state_dim)
    for (j, b) in enumerate(vcat(lb, ub))
        i = (j - 1) % state_dim + 1
        _lb = vcat(lb[1:i-1], lb[i+1:end])
        _ub = vcat(ub[1:i-1], ub[i+1:end])
        V_boundary = (state, p) -> V(vcat(state[1:i-1], b, state[i:end]))
        function ∇V_boundary(state, p)
            g = ∇V(vcat(state[1:i-1], b, state[i:end]))
            return vcat(g[1:i-1], g[i+1:end])
        end
        f = OptimizationFunction(
            V_boundary,
            Optimization.AutoForwardDiff(),
            grad = ∇V_boundary,
        )
        state0 = (_lb + _ub) / 2
        prob = OptimizationProblem(f, state0, lb = _lb, ub = _ub)
        opt = OptimizationOptimJL.ParticleSwarm(lower = _lb, upper = _ub, n_particles = 100)
        res = solve(prob, opt)
        candidates[j] = vcat(res.u[1:i-1], b, res.u[i:end])
        V(candidates[j])
    end
    V_min, j_opt = findmin(V, candidates)

    # Find a point just interior of the boundary to start optimization
    i_bd = (j_opt - 1) % state_dim + 1
    x_min = candidates[j_opt]

    return (V_min, x_min, i_bd)
end

function make_paramerized_scalar_func(f)
    function f_param(res, x, p)
        res .= f(x)
    end
    function f_param(x, p)
        f(x)
    end
    return f_param
end

function make_paramerized_vector_func(f)
    function f_param(res, x, p)
        res .= reshape(f(x), size(res))
    end
    function f_param(x, p)
        f(x)
    end
    return f_param
end

"""
    get_RoA_estimate_aided(V, dVdt, lb, ub, V_certified, ρ_certified; fixed_point, ∇V,
                           ∇V_certified)

Finds the level of the largest sublevelset comletely in the domain in which the
Lyapunov conditions are met, ignoring points `x` where `V_certified(x) < ρ_certified`.

Specifically, finds the largest `ρ` such that
    `V(x) < ρ => lb .< x .< ub && (dVdt(x) < 0 || V_certified(x) < ρ_certified)`
If `∇V` is not specified, it is calculated using `ForwardDiff`. If `fixed_point` is
not specified, it defaults to the origin, with dimension inferred from `lb`.
"""
function get_RoA_estimate_aided(V, dVdt, lb, ub, V_certified, ρ_certified; reltol = 1e-3,
                                abstol = 1e-4, fixed_point = zeros(length(lb)),
                                ∇V = nothing, ∇V_certified = nothing)
    state_dim = length(lb)
    if isnothing(∇V)
        ∇V_func(state::AbstractVector) = ForwardDiff.gradient(V, state)
        ∇V_func(states::AbstractMatrix) = mapslices(∇V_func, states, dims = [1])
    else
        ∇V_func = ∇V
    end
    if isnothing(∇V_certified)
        ∇V_certified_func(state::AbstractVector) = ForwardDiff.gradient(V, state)
        ∇V_certified_func(states::AbstractMatrix) = mapslices(∇V_func, states, dims = [1])
    else
        ∇V_certified_func = ∇V_certified
    end
    _get_RoA_estimate_aided(V, dVdt, state_dim, lb, ub, V_certified, ρ_certified,
                            fixed_point, ∇V_func, ∇V_certified_func, reltol, abstol)
end

function _get_RoA_estimate_aided(V, dVdt, state_dim, lb, ub, V_certified, ρ_certified,
                                 fixed_point, ∇V, ∇V_certified, reltol, abstol)
    # Let ρ_max = minimum value of V on the boundary
    ρ_max, x_opt, i_bd = find_min_on_bounding_box(V, state_dim, lb, ub, ∇V)

    # Move x_opt just off the boundary
    x_opt[i_bd] -= √eps(typeof(x_opt[i_bd])) * sign(x_opt[i_bd] - fixed_point[i_bd])

    # Set up paramerized versions of functions for optimization
    negV̇_param = make_paramerized_scalar_func(x -> -dVdt(x))
    cons = make_paramerized_vector_func(x -> vcat(V(x), V_certified(x)))
    cons_j = make_paramerized_vector_func(x -> vcat(reshape(∇V(x), (1,length(x))),
                                                    reshape(∇V_certified(x), (1,length(x)))))
    V_param = make_paramerized_scalar_func(V)
    ∇V_param = make_paramerized_vector_func(∇V)
    V_cert_param = make_paramerized_scalar_func(V_certified)
    ∇V_cert_param = make_paramerized_vector_func(∇V_certified)

    # Initialize search for:
    #   max ρ : ( (max V̇(x) : (V(x) < ρ && V_certified > ρ_certified)) < 0)
    # We cannot verify V(x) > ρ_max, since some such x are outside the domain
    # We have also (by assumption) verified {x : V_certified(x) < ρ_certified},
    # so we have also verified {x : V(x) < ρ_min}, where
    #   ρ_min = min V(x) : V_certified(x) ≥ ρ_certified
    # since V(x) < ρ_min => V_certified(x) < ρ_certified
    ρ_min, x_min = find_min_on_region_boundary(
        V_param,
        V_cert_param,
        ρ_certified,
        lb,
        ub,
        ∇V_param,
        ∇V_cert_param
    )

    if ρ_min > ρ_max
        @show ρ_min, ρ_max, x_min
        throw("Something wen't wrong")
    end

    # Our search is for ρ ∈ [ρ_min, ρ_max] and we begin by attempting to
    # validate the whole set {x : V(x) < ρ_max}
    ρ = ρ_max

    # Set up OptimizationFunction
    f = OptimizationFunction{true}(
        negV̇_param,
        Optimization.AutoFiniteDiff();
        cons = cons,
        cons_j = cons_j,
    )

    # Stop the search early if we find any x : V̇(x) > 0
    cb = (x, negdVdt) -> negdVdt < 0 # i.e., V̇ > 0

    # Binary search
    global count = 1
    while abs(ρ_max - ρ_min) > max(abstol, reltol*ρ_max)
        @show count
        # Find a point just interior of the boundary to start optimization
        guess = initialize_guess_aided(
            V_param,
            cons,
            ρ_certified,
            lb,
            ub,
            (ρ_min + ρ) / 2, # addresses trouble with V(guess) < ρ_min
            ρ,
            x_opt,
            ∇V_param,
            cons_j
        )

        # Find max V̇(x) : ρ_min ≤ V(x) < ρ, V_certified(x) > ρ_certified
        # We've already verified {x : V(x) < ρ_min || V_certified(x) < ρ_certified}
        # so don't need to check there
        prob = OptimizationProblem{true}(
            f,
            guess,
            lb = lb,
            ub = ub,
            lcons = [ρ_min, ρ_certified],
            ucons = [ρ, Inf],
            callback = cb,
        )
        opt = OptimizationOptimJL.IPNewton()
        res = solve(prob, opt, allow_f_increases = true, successive_f_tol = 2)
        V̇_max = dVdt(res.u)

        if V̇_max > √eps(Float64)
            # Since V̇(x_opt) > 0, we cannot verify any x : V(x) ≥ V(x_opt),
            # so we limit our search to ρ ∈ [ρ_min, V(x_opt)]
            x_opt = res.u
            ρ_max = V(x_opt)
        else
            # Since (max V̇(x) : V(x) < ρ) < 0, we have verified {x : V(x) < ρ}
            # We continue our search of [ρ, ρ_max] to see if we can do better
            ρ_min = ρ
        end
        ρ = (ρ_max + ρ_min) / 2
        global count += 1
    end
    return ρ_min
end

"""
    find_min_on_region_boundary(V, boundary_func, boundary_val, lb, ub, ∇V, ∇boundary_func)

Finds min V(x) : boundary_func(x) ≥ boundary_val && lb .< x .< ub
"""
function find_min_on_region_boundary(V, boundary_func, boundary_val, lb, ub, ∇V,
                                     ∇boundary_func)
    function neg_bf(x, p)
        -boundary_func(x,p)
    end
    function neg_bf(res, x, p)
        res .= -boundary_func(x,p)
    end
    function neg_∇bf(x, p)
        -∇boundary_func(x,p)
    end
    function neg_∇bf(res, x, p)
        res .= reshape(-∇boundary_func(x,p), size(res))
    end

    # Initialize guess by finding x : boundary_func(x) ≥ boundary_val
    # We do so by solving max boundary_func(x) = min -boundary_func(x) and
    # stopping early if boundary_func(x) ≥ boundary_val + ϵ
    f_guess = OptimizationFunction{true}(
        neg_bf,
        Optimization.AutoFiniteDiff();
        grad = neg_∇bf,
    )
    prob_guess = OptimizationProblem{true}(
        f_guess,
        (lb + ub) / 2, # We only need a point on the interior of the bounding box
        lb = lb .+ √eps(typeof(first(lb))),
        ub = ub .- √eps(typeof(first(ub))),
    )
    opt = OptimizationOptimJL.ParticleSwarm()
    cb = (x, neg_bd_f) -> neg_bd_f ≤ -boundary_val - √eps(typeof(boundary_val))
    res = solve(prob_guess, opt, allow_f_increases = true, successive_f_tol = 2, callback=cb)
    guess = res.u

    if boundary_func(guess, nothing) < boundary_val
        @show guess, boundary_func(guess, nothing), boundary_val
        throw("Generated bad initial guess")
    end
    #=
    # Find x : boundary_func(x) == boundary_val + ϵ
    function boundary_classifier(x,p)
        boundary_func(x,p) - boundary_val - √eps(typeof(boundary_val))
    end
    function boundary_classifier(res, x, p)
        res .= boundary_func(x,p) - boundary_val - √eps(typeof(boundary_val))
    end
    f_guess = NonlinearFunction(boundary_classifier)
    prob_guess = NonlinearProblem(f_guess, guess)
    res = solve(prob_guess, NewtonRaphson())
    guess = res.u
    @show guess
    =#

    # Find min V(x) : boundary_func(x) ≥ boundary_val
    f = OptimizationFunction{true}(
        V,
        Optimization.AutoFiniteDiff();
        grad = ∇V,
        cons = boundary_func,
        cons_j = ∇boundary_func,
    )
    prob = OptimizationProblem{true}(
        f,
        guess, # guaranteed by the above optimization to satisfy the constraints
        lb = lb,
        ub = ub,
        lcons = [boundary_val],
        ucons = [boundary_func(guess,nothing)*1.1],
    )
    opt = OptimizationOptimJL.IPNewton()
    res = solve(prob, opt, allow_f_increases = true, successive_f_tol = 2)
    return V(res.u, nothing), res.u
end
