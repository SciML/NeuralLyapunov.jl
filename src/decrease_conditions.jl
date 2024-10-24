"""
    LyapunovDecreaseCondition(check_decrease, rate_metric, strength, rectifier)

Specifies the form of the Lyapunov decrease condition to be used.

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

# Training conditions

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

This can also be accomplished with [`AsymptoticStability`](@ref).

Exponential decrease of rate ``k`` is proven by
    ``V̇(x) ≤ - k V(x)``,
which corresponds to

    rate_metric = (V, dVdt) -> dVdt + k * V
    strength = (x, x0) -> 0.0

This can also be accomplished with [`ExponentialStability`](@ref).

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
    StabilityISL(; rectifier)

Construct a [`LyapunovDecreaseCondition`](@ref) corresponding to stability in the sense of
Lyapunov (i.s.L.).

Stability i.s.L. is proven by ``V̇(x) ≤ 0``. The inequality is represented by
``\\texttt{rectifier}(V̇(x)) = 0``. The default value `rectifier = (t) -> max(zero(t), t)`
exactly represents the inequality, but differentiable approximations of this function may be
employed.
"""
function StabilityISL(; rectifier = (t) -> log(one(t) + exp(t))) # max(zero(t), t)) #
    return LyapunovDecreaseCondition(
        true,
        (V, dVdt) -> dVdt,
        (x, x0) -> 0.0,
        rectifier
    )
end

"""
    AsymptoticStability(; C, rectifier)

Construct a [`LyapunovDecreaseCondition`](@ref) corresponding to asymptotic stability.

The decrease condition for asymptotic stability is ``V̇(x) < 0``, which is here represented
as ``V̇(x) ≤ - C \\lVert x - x_0 \\rVert^2`` for some ``C > 0``. ``C`` defaults to `1e-6`.

The inequality is represented by
``\\texttt{rectifier}(V̇(x) + C \\lVert x - x_0 \\rVert^2) = 0``.
The default value `rectifier = (t) -> max(zero(t), t)` exactly represents the inequality,
but differentiable approximations of this function may be employed.
"""
function AsymptoticStability(;
        C::Real = 1e-6,
        rectifier = (t) -> max(zero(t), t)
)::LyapunovDecreaseCondition
    return LyapunovDecreaseCondition(
        true,
        (V, dVdt) -> dVdt,
        (x, x0) -> C * (x - x0) ⋅ (x - x0),
        rectifier
    )
end

"""
    ExponentialStability(k; rectifier)

Construct a [`LyapunovDecreaseCondition`](@ref) corresponding to exponential stability of
rate ``k``.

The Lyapunov condition for exponential stability is ``V̇(x) ≤ -k V(x)`` for some ``k > 0``.

The inequality is represented by ``\\texttt{rectifier}(V̇(x) + k V(x)) = 0``.
The default value `rectifier = (t) -> max(zero(t), t)` exactly represents the inequality,
but differentiable approximations of this function may be employed.
"""
function ExponentialStability(
        k::Real;
        rectifier = (t) -> max(zero(t), t)
)::LyapunovDecreaseCondition
    return LyapunovDecreaseCondition(
        true,
        (V, dVdt) -> dVdt + k * V,
        (x, x0) -> 0.0,
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
