"""
    get_RoA_estimate(V, dVdt, lb, ub)

Finds the level of the largest sublevelset comletely in the domain in which the
Lyapunov conditions are met. Specifically finds the largest ρ such that
    V(x) < ρ => lb .< x .< ub && dVdt(x) < 0
"""
function get_RoA_estimate(V, dVdt, lb, ub; fixed_point = nothing, ∇V = nothing)
    state_dim = length(lb)

    fixed_point = isnothing(fixed_point) ? zeros(state_dim) : fixed_point

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
        @show candidates[j] = vcat(res.u[1:i-1], b, res.u[i:end])
        @show V(candidates[j])
    end
    @show ρ_max, j_guess = findmin(V, candidates)

    # Find a point just interior of the boundary to start optimization
    guess = candidates[j_guess]
    i_bd = (j_guess - 1) % state_dim + 1
    guess[i_bd] = 0.9 * (guess[i_bd] - fixed_point[i_bd]) + fixed_point[i_bd]

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
        ∇V_out .= transpose(∇V(x))
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

    while abs(ρ_max - ρ_min) > √eps(Float64)
        # Find max V̇(x) : ρ_min ≤ V(x) < ρ_max
        # Since, we've already verified V(x) < ρ_min and excluded V(x) > ρ_max
        prob = OptimizationProblem{true}(
            f,
            guess,
            lb = lb,
            ub = ub,
            lcons = [ρ_min],
            ucons = [ρ],
        )
        opt = OptimizationOptimJL.IPNewton()
        res = solve(prob, opt, allow_f_increases = true, successive_f_tol = 2)
        V̇_max = dVdt(res.u)

        if V̇_max > √eps(Float64)
            ρ_max = V(res.u)
            guess = 0.9 * (res.u - fixed_point) + fixed_point
        else
            ρ_min = ρ
        end
        ρ = (ρ_max + ρ_min) / 2
    end
    return ρ_min
end
