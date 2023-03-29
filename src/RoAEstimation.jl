"""
    get_RoA_estimate(V, dVdt, lb, ub)

Finds the level of the largest sublevelset comletely in the domain in which the
Lyapunov conditions are met. Specifically finds the largest ρ such that
    V(x) < ρ => lb .< x .< ub && dVdt(x) < 0
"""
function get_RoA_estimate(V, dVdt, lb, ub; fixed_point = nothing, ∇V = nothing)
    state_dim = length(lb)
    fixed_point = isnothing(fixed_point) ? zeros(state_dim) : fixed_point
    get_RoA_estimate(V, dVdt, state_dim, lb, ub, fixed_point, ∇V)
end

function get_RoA_estimate(V, dVdt, state_dim, lb, ub, fixed_point, ∇V)
    # Let ρ_max = minimum value of V on the boundary
    # TODO: @view to speed up?
    candidates = Vector{Any}(undef, 2 * state_dim)
    for (j, b) in enumerate(vcat(lb, ub))
        i = (j - 1) % state_dim + 1
        _lb = vcat(lb[1:i-1], lb[i+1:end])
        _ub = vcat(ub[1:i-1], ub[i+1:end])
        V_boundary = (state, p) -> V(vcat(state[1:i-1], b, state[i:end]))
        ∇V_boundary = if !isnothing(∇V)
            function (state, p)
                g = ∇V(vcat(state[1:i-1], b, state[i:end]))
                return vcat(g[1:i-1], g[i+1:end])
            end
        else
            nothing
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
    ρ_max, j_opt = findmin(V, candidates)
    i_bd = (j_opt - 1) % state_dim + 1
    x_opt = candidates[j_opt]
    x_opt[i_bd] -= √eps(typeof(x_opt[i_bd])) * sign(x_opt[i_bd] - fixed_point[i_bd]) 

    # Binary search for max ρ : ( (max V̇(x) : V(x) < ρ) < 0)
    ρ_min = 0.0
    ρ = ρ_max

    function negV̇(dV, x, p)
        dV .= -dVdt(x)
    end
    function negV̇(x, p)
        -dVdt(x)
    end
    function V_param(V_out, x, p)
        V_out .= V(x)
    end
    function V_param(x, p)
        V(x)
    end
    function ∇V_param(∇V_out, x, p)
        ∇V_out .=  reshape(∇V(x), size(∇V_out))
    end
    function ∇V_param(x, p)
        ∇V(x)
    end

    f = OptimizationFunction{true}(
        negV̇,
        Optimization.AutoFiniteDiff();
        cons = V_param,
        cons_j = ∇V_param,
    )
    cb = (x, negdVdt) -> negdVdt < 0 # i.e., V̇ > 0

    while abs(ρ_max - ρ_min) > √eps(typeof(ρ_min))
        # Find a point just interior of the boundary to start optimization
        guess = initialize_guess(V_param, lb, ub, ρ_min, ρ, x_opt; ∇V_param)
        # Find max V̇(x) : ρ_min ≤ V(x) < ρ_max
        # Since, we've already verified V(x) < ρ_min and excluded V(x) > ρ_max
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
            x_opt = res.u
            ρ_max = V(x_opt)
        else
            ρ_min = ρ
        end
        ρ = (ρ_max + ρ_min) / 2
    end
    return ρ_min
end

"""
    initialize_guess(V_param, lb, ub, ρ_min, ρ, x_opt; ∇V_param = nothing)
Finds a point x of the same shape as x_opt such that ρ_min < V(x) < ρ
"""
function initialize_guess(V_param, lb, ub, ρ_min, ρ, x_opt; ∇V_param = nothing)
    f = OptimizationFunction{true}(
        V_param,
        Optimization.AutoFiniteDiff();
        grad = ∇V_param,
        cons = V_param,
        cons_j = ∇V_param,
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
