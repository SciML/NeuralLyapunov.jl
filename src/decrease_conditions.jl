"""
    LyapunovDecreaseCondition(check_decrease, rate_metric, strength, rectifier)

Specifies the form of the Lyapunov conditions to be used.

# Fields
  - `check_decrease::Bool`: whether or not to train for negativity/nonpositivity of
    ``V̇(x)``.
  - `rate_metric::Function`: should increase with ``V̇(x)``; used when
    `check_decrease == true`.
  - `strength::Function`: specifies the level of strictness for negativity training; should
    be zero when the two inputs are equal and nonnegative otherwise; used when
    `check_decrease == true`.
  - `rectifier::Function`: positive when the input is positive and (approximately) zero when
    the input is negative.

If `check_decrease == true`, training will enforce:

``\\texttt{rate\\_metric}(V(x), V̇(x)) ≤ -\\texttt{strength}(x, x_0).``

The inequality will be approximated by the equation:

``\\texttt{rectifier}(\\texttt{rate\\_metric}(V(x), V̇(x)) + \\texttt{strength}(x, x_0)) = 0.``

Note that the approximate equation and inequality are identical when
``\\texttt{rectifier}(t) = \\max(0, t)``.

If the dynamics truly have a fixed point at ``x_0`` and ``V̇(x)`` is truly the rate of
decrease of ``V(x)`` along the dynamics, then ``V̇(x_0)`` will be ``0`` and there is no need
to train for ``V̇(x_0) = 0``.

# Examples:

Asymptotic decrease can be enforced by requiring
    ``V̇(x) ≤ -C \\lVert x - x_0 \\rVert^2``,
for some positive ``C``, which corresponds to

    rate_metric = (V, dVdt) -> dVdt
    strength = (x, x0) -> C * (x - x0) ⋅ (x - x0)

This can also be accomplished with [`AsymptoticDecrease`](@ref).

Exponential decrease of rate ``k`` is proven by
    ``V̇(x) ≤ - k * V(x)``,
which corresponds to

    rate_metric = (V, dVdt) -> dVdt + k * V
    strength = (x, x0) -> 0.0

This can also be accomplished with [`ExponentialDecrease`](@ref).


In either case, the rectified linear unit `rectifier = (t) -> max(zero(t), t)` exactly
represents the inequality, but differentiable approximations of this function may be
employed.
"""
struct LyapunovDecreaseCondition <: AbstractLyapunovDecreaseCondition
    check_decrease::Bool
    rate_metric::Function
    strength::Function
    rectifier::Function
end

function check_decrease(cond::LyapunovDecreaseCondition)::Bool
    cond.check_decrease
end

function get_decrease_condition(cond::LyapunovDecreaseCondition)
    if cond.check_decrease
        return (V, dVdt, x, fixed_point) -> cond.rectifier(
            cond.rate_metric(V(x), dVdt(x)) + cond.strength(x, fixed_point)
        )
    else
        return nothing
    end
end

"""
    AsymptoticDecrease(; strict, C, rectifier)

Construct a [`LyapunovDecreaseCondition`](@ref) corresponding to asymptotic decrease.

If `strict == false`, the decrease condition is
``\\dot{V}(x) ≤ 0``,
and if `strict == true`, the condition is
``\\dot{V}(x) ≤ - C \\lVert x - x_0 \\rVert^2``.

The inequality is represented by
``\\texttt{rectifier}(\\dot{V}(x) + C \\lVert x - x_0 \\rVert^2) = 0``.
"""
function AsymptoticDecrease(;
        strict::Bool = false,
        C::Real = 1e-6,
        rectifier = (t) -> max(zero(t), t)
)::LyapunovDecreaseCondition
    strength = if strict
        (x, x0) -> C * (x - x0) ⋅ (x - x0)
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

Construct a [`LyapunovDecreaseCondition`](@ref) corresponding to exponential decrease of
rate ``k``.

If `strict == false`, the condition is ``\\dot{V}(x) ≤ -k V(x)``, and if `strict == true`,
the condition is ``\\dot{V}(x) ≤ -k V(x) - C \\lVert x - x_0 \\rVert^2``.

The inequality is represented by
``\\texttt{rectifier}(\\dot{V}(x) + k V(x) + C \\lVert x - x_0 \\rVert^2) = 0``.
"""
function ExponentialDecrease(
        k::Real;
        strict::Bool = false,
        C::Real = 1e-6,
        rectifier = (t) -> max(zero(t), t)
)::LyapunovDecreaseCondition
    strength = if strict
        (x, x0) -> C * (x - x0) ⋅ (x - x0)
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

Construct a [`LyapunovDecreaseCondition`](@ref) which represents not checking for
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
