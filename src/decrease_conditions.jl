"""
    LyapunovDecreaseCondition(check_decrease, decrease, strength, relu, check_fixed_point)

Specifies the form of the Lyapunov conditions to be used; if `check_decrease`, training will
enforce `decrease(V, dVdt) ≤ strength(state, fixed_point)`.

The inequality will be approximated by the equation
    `relu(decrease(V, dVdt) - strength(state, fixed_point)) = 0.0`.
If `check_fixed_point` is `false`, then training assumes `dVdt(fixed_point) = 0`, but
if `check_fixed_point` is `true`, then training will enforce `dVdt(fixed_point) = 0`.

If the dynamics truly have a fixed point at `fixed_point` and `dVdt` has been defined
properly in terms of the dynamics, then `dVdt(fixed_point)` will be `0` and there is no need
to enforce `dVdt(fixed_point) = 0`, so `check_fixed_point` defaults to `false`.

# Examples:

Asymptotic decrease can be enforced by requiring
    `dVdt ≤ -C |state - fixed_point|^2`,
which corresponds to
    `decrease = (V, dVdt) -> dVdt`
    `strength = (x, x0) -> -C * (x - x0) ⋅ (x - x0)`

Exponential decrease of rate `k` is proven by `dVdt ≤ - k * V`, so corresponds to
    `decrease = (V, dVdt) -> dVdt + k * V`
    `strength = (x, x0) -> 0.0`
"""
struct LyapunovDecreaseCondition <: AbstractLyapunovDecreaseCondition
    check_decrease::Bool
    decrease::Function
    strength::Function
    relu::Function
    check_fixed_point::Bool
end

function check_decrease(cond::LyapunovDecreaseCondition)::Bool
    cond.check_decrease
end

function check_stationary_fixed_point(cond::LyapunovDecreaseCondition)::Bool
    cond.check_fixed_point
end

function get_decrease_condition(cond::LyapunovDecreaseCondition)
    if cond.check_decrease
        return (V, dVdt, x, fixed_point) -> cond.relu(
            cond.decrease(V(x), dVdt(x)) - cond.strength(x, fixed_point)
        )
    else
        return nothing
    end
end

"""
    AsymptoticDecrease(strict; check_fixed_point, C)

Constructs a `LyapunovDecreaseCondition` corresponding to asymptotic decrease.

If `strict` is `false`, the condition is `dV/dt ≤ 0`, and if `strict` is `true`, the
condition is `dV/dt ≤ - C | state - fixed_point |^2`.

The inequality is represented by `a ≥ b` <==> `relu(b-a) = 0.0`.
"""
function AsymptoticDecrease(;
        strict::Bool = false,
        check_fixed_point::Bool = false,
        C::Real = 1e-6,
        relu = (t) -> max(0.0, t)
)::LyapunovDecreaseCondition
    strength = if strict
            (x, x0) -> -C * (x - x0) ⋅ (x - x0)
    else
            (x, x0) -> 0.0
    end

    return LyapunovDecreaseCondition(
            true,
            (V, dVdt) -> dVdt,
            strength,
            relu,
            check_fixed_point
        )
end

"""
    ExponentialDecrease(k, strict; check_fixed_point, C)

Constructs a `LyapunovDecreaseCondition` corresponding to exponential decrease of rate `k`.

If `strict` is `false`, the condition is `dV/dt ≤ -k * V`, and if `strict` is `true`, the
condition is `dV/dt ≤ -k * V - C * ||state - fixed_point||^2`.

The inequality is represented by `a ≥ b` <==> `relu(b-a) = 0.0`.
"""
function ExponentialDecrease(
        k::Real;
        strict::Bool = false,
        check_fixed_point::Bool = false,
        C::Real = 1e-6,
        relu = (t) -> max(0.0, t)
)::LyapunovDecreaseCondition
    strength = if strict
        (x, x0) -> -C * (x - x0) ⋅ (x - x0)
    else
        (x, x0) -> 0.0
    end

    return LyapunovDecreaseCondition(
            true,
            (V, dVdt) -> dVdt + k * V,
            strength,
            relu,
            check_fixed_point
        )
end

"""
    DontCheckDecrease(check_fixed_point = false)

Constructs a `LyapunovDecreaseCondition` which represents not checking for
decrease of the Lyapunov function along system trajectories. This is appropriate
in cases when the Lyapunov decrease condition has been structurally enforced.

It is still possible to check for `dV/dt = 0` at `fixed_point`, even in this case.
"""
function DontCheckDecrease(check_fixed_point::Bool = false)::LyapunovDecreaseCondition
    return LyapunovDecreaseCondition(
        false,
        (V, dVdt) -> 0.0,
        (state, fixed_point) -> 0.0,
        (t) -> 0.0,
        check_fixed_point
    )
end
