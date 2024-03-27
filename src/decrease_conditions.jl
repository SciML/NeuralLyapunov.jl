"""
    LyapunovDecreaseCondition(check_decrease, decrease, strength, rectifier)

Specifies the form of the Lyapunov conditions to be used; if `check_decrease`, training will
enforce `decrease(V, dVdt) ≤ strength(state, fixed_point)`.

The inequality will be approximated by the equation
    `rectifier(decrease(V, dVdt) - strength(state, fixed_point)) = 0.0`.

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
    rectifier::Function
end

function check_decrease(cond::LyapunovDecreaseCondition)::Bool
    cond.check_decrease
end

function get_decrease_condition(cond::LyapunovDecreaseCondition)
    if cond.check_decrease
        return (V, dVdt, x, fixed_point) -> cond.rectifier(
            cond.decrease(V(x), dVdt(x)) - cond.strength(x, fixed_point)
        )
    else
        return nothing
    end
end

"""
    AsymptoticDecrease(; strict, C, rectifier)

Construct a `LyapunovDecreaseCondition` corresponding to asymptotic decrease.

If `strict` is `false`, the condition is `dV/dt ≤ 0`, and if `strict` is `true`, the
condition is `dV/dt ≤ - C | state - fixed_point |^2`.

The inequality is represented by `a ≥ b` <==> `rectifier(b-a) = 0.0`.
"""
function AsymptoticDecrease(;
        strict::Bool = false,
        C::Real = 1e-6,
        rectifier = (t) -> max(zero(t), t)
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
        rectifier
    )
end

"""
    ExponentialDecrease(k; strict, C, rectifier)

Construct a `LyapunovDecreaseCondition` corresponding to exponential decrease of rate `k`.

If `strict` is `false`, the condition is `dV/dt ≤ -k * V`, and if `strict` is `true`, the
condition is `dV/dt ≤ -k * V - C * ||state - fixed_point||^2`.

The inequality is represented by `a ≥ b` <==> `rectifier(b-a) = 0.0`.
"""
function ExponentialDecrease(
        k::Real;
        strict::Bool = false,
        C::Real = 1e-6,
        rectifier = (t) -> max(zero(t), t)
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
        rectifier
    )
end

"""
    DontCheckDecrease()

Construct a `LyapunovDecreaseCondition` which represents not checking for
decrease of the Lyapunov function along system trajectories. This is appropriate
in cases when the Lyapunov decrease condition has been structurally enforced.
"""
function DontCheckDecrease()::LyapunovDecreaseCondition
    return LyapunovDecreaseCondition(
        false,
        (V, dVdt) -> zero(V),
        (state, fixed_point) -> 0.0,
        (t) -> zero(t)
    )
end
