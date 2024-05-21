"""
    LyapunovMinimizationCondition(check_nonnegativity, strength, rectifier, check_fixed_point)

Specifies the form of the Lyapunov minimization condition to be used.

# Fields
  - `check_nonnegativity::Bool`: whether or not to train for positivity/non-negativity of ``V(x)``
  - `strength::Function`: level of strictness for positivity training, if `check_nonnegativity == true`
  - `rectifier::Function`: positive when the input is positive and (approximately) zero when the input is negative
  - `check_fixed_point`: whether or not to train for ``V(x_0) = 0``.

# Training conditions

If `check_nonnegativity` is `true`, training will attempt to enforce
    ``V(x) ≥ \\texttt{strength}(x, x_0)``.
The inequality will be approximated by the equation
    ``\\texttt{rectifier}(\\texttt{strength}(x, x_0) - V(x_0)) = 0``.

If `check_fixed_point` is `true`, then training will also attempt to enforce
    ``V(x_0) = 0``.

# Examples

When training for a strictly positive definite ``V``, an example of an appropriate `strength`
is ``\\texttt{strength}(x, x_0) = \\lVert x - x_0 \\rVert^2``.
This form is used in [`StrictlyPositiveDefinite`](@ref).

If ``V`` were structured such that it is always nonnegative, then ``V(x_0) = 0`` is all
that must be enforced in training for the Lyapunov function to be uniquely minimized at
``x_0``. So, in that case, we would use
    `check_nonnegativity = false;  check_fixed_point = true`.
This can also be accomplished with [`DontCheckNonnegativity(true)`](@ref).

In either case, `rectifier = (t) -> max(0.0, t)` exactly represents the inequality, but
differentiable approximations of this function may be employed.
"""
struct LyapunovMinimizationCondition <: AbstractLyapunovMinimizationCondition
    check_nonnegativity::Bool
    strength::Function
    rectifier::Function
    check_fixed_point::Bool
end

function check_nonnegativity(cond::LyapunovMinimizationCondition)::Bool
    cond.check_nonnegativity
end

function check_minimal_fixed_point(cond::LyapunovMinimizationCondition)::Bool
    cond.check_fixed_point
end

function get_minimization_condition(cond::LyapunovMinimizationCondition)
    if cond.check_nonnegativity
        return (V, x, fixed_point) -> cond.rectifier(cond.strength(x, fixed_point) - V(x))
    else
        return nothing
    end
end

"""
    StrictlyPositiveDefinite(C; check_fixed_point, rectifier)

Construct a `LyapunovMinimizationCondition` representing
    `V(state) ≥ C * ||state - fixed_point||^2`.
If `check_fixed_point` is `true`, then training will also attempt to enforce
    `V(fixed_point) = 0`.

The inequality is represented by `a ≥ b` <==> `rectifier(b-a) = 0.0`.
"""
function StrictlyPositiveDefinite(;
        check_fixed_point = true,
        C::Real = 1e-6,
        rectifier = (t) -> max(zero(t), t)
)::LyapunovMinimizationCondition
    LyapunovMinimizationCondition(
        true,
        (state, fixed_point) -> C * (state - fixed_point) ⋅ (state - fixed_point),
        rectifier,
        check_fixed_point
    )
end

"""
    PositiveSemiDefinite(check_fixed_point)

Construct a `LyapunovMinimizationCondition` representing
    `V(state) ≥ 0`.
If `check_fixed_point` is `true`, then training will also attempt to enforce
    `V(fixed_point) = 0`.

The inequality is represented by `a ≥ b` <==> `rectifier(b-a) = 0.0`.
"""
function PositiveSemiDefinite(;
        check_fixed_point = true,
        rectifier = (t) -> max(zero(t), t)
)::LyapunovMinimizationCondition
    LyapunovMinimizationCondition(
        true,
        (state, fixed_point) -> 0.0,
        rectifier,
        check_fixed_point
    )
end

"""
    DontCheckNonnegativity(check_fixed_point)

Construct a `LyapunovMinimizationCondition` which represents not checking for nonnegativity
of the Lyapunov function. This is appropriate in cases where this condition has been
structurally enforced.

It is still possible to check for `V(fixed_point) = 0`, even in this case, for example if
`V` is structured to be positive for `state ≠ fixed_point`, but it is not guaranteed
structurally that `V(fixed_point) = 0`.
"""
function DontCheckNonnegativity(; check_fixed_point = false)::LyapunovMinimizationCondition
    LyapunovMinimizationCondition(
        false,
        (state, fixed_point) -> 0.0,
        (t) -> zero(t),
        check_fixed_point
    )
end
