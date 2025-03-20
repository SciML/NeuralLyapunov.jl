"""
    RoAAwareDecreaseCondition(cond, sigmoid, ρ, out_of_RoA_penalty)

Specifies the form of the Lyapunov decrease condition to be used, training for a region of
attraction estimate of ``\\{ x : V(x) ≤ ρ \\}``.

# Fields
  - `cond::AbstractLyapunovDecreaseCondition`: specifies the loss to be applied when
    ``V(x) ≤ ρ``.
  - `sigmoid::Function`: approximately one when the input is positive and approximately zero
    when the input is negative.
  - `ρ`: the level of the sublevel set forming the estimate of the region of attraction.
  - `out_of_RoA_penalty::Function`: a loss function to be applied penalizing points outside
    the sublevel set forming the region of attraction estimate.

# Training conditions

Inside the region of attraction (i.e., when ``V(x) ≤ ρ``), the loss function specified by
`cond` will be applied. Outside the region of attraction (i.e., when ``V(x) > ρ``), the loss
function specified by `out_of_RoA_penalty` will be applied.

The `sigmoid` function allows for a smooth transition between the ``V(x) ≤ ρ`` case and the
``V(x) > ρ`` case, by combining the different losses into one:

``\\texttt{sigmoid}(ρ - V(x)) (\\text{in-RoA expression}) + \\texttt{sigmoid}(V(x) - ρ) (\\text{out-of-RoA expression}) = 0``.

Note that a hard transition, which only enforces the in-RoA equation when ``V(x) ≤ ρ`` and
the out-of-RoA equation when ``V(x) > ρ`` can be provided by a `sigmoid` which is exactly
one when the input is nonnegative and exactly zero when the input is negative.

If the dynamics truly have a fixed point at ``x_0`` and ``V̇(x)`` is truly the rate of
decrease of ``V(x)`` along the dynamics, then ``V̇(x_0)`` will be ``0`` and there is no need
to train for ``V̇(x_0) = 0``.

# Examples:

Asymptotic decrease can be enforced by requiring
    ``V̇(x) ≤ -C \\lVert x - x_0 \\rVert^2``,
for some positive ``C``, which can be represented by a [`LyapunovDecreaseCondition`](@ref)
with

    rate_metric = (V, dVdt) -> dVdt
    strength = (x, x0) -> C * (x - x0) ⋅ (x - x0)

Exponential decrease of rate ``k`` is proven by
    ``V̇(x) ≤ - k V(x)``,
which can be represented by a [`LyapunovDecreaseCondition`](@ref) with

    rate_metric = (V, dVdt) -> dVdt + k * V
    strength = (x, x0) -> 0.0

Enforcing either condition only in the region of attraction and not penalizing any points
outside that region would correspond to

    out_of_RoA_penalty = (V, dVdt, state, fixed_point, ρ) -> 0.0

whereas an example of a penalty that decays farther in state space from the fixed point is

    out_of_RoA_penalty = (V, dVdt, x, x0, ρ) -> 1.0 / ((x - x0) ⋅ (x - x0))

Note that this penalty could also depend on values of ``V`` and ``V̇`` at various points, as
well as ``ρ``.

In any of these cases, the rectified linear unit `rectifier = (t) -> max(zero(t), t)`
exactly represents the inequality, but differentiable approximations of this function may be
employed.

See also: [`AbstractLyapunovDecreaseCondition`](@ref), [`LyapunovDecreaseCondition`](@ref)
"""
struct RoAAwareDecreaseCondition <: AbstractLyapunovDecreaseCondition
    cond::AbstractLyapunovDecreaseCondition
    sigmoid::Function
    ρ::Real
    out_of_RoA_penalty::Function
end

check_decrease(cond::RoAAwareDecreaseCondition)::Bool = check_decrease(cond.cond)

function get_decrease_condition(cond::RoAAwareDecreaseCondition)
    if check_decrease(cond)
        in_RoA_penalty = get_decrease_condition(cond.cond)
        return function (V, dVdt, x, fixed_point)
            _V = V(x)
            _V = _V isa AbstractVector ? _V[] : _V
            _V̇ = dVdt(x)
            _V̇ = _V̇ isa AbstractVector ? _V̇[] : _V̇
            return cond.sigmoid(cond.ρ - _V) * in_RoA_penalty(V, dVdt, x, fixed_point) +
                   cond.sigmoid(_V - cond.ρ) *
                   cond.out_of_RoA_penalty(_V, _V̇, x, fixed_point, cond.ρ)
        end
    else
        return nothing
    end
end

"""
    make_RoA_aware(cond; ρ, out_of_RoA_penalty, sigmoid)

Add awareness of the region of attraction (RoA) estimation task to the supplied
[`AbstractLyapunovDecreaseCondition`](@ref).

When estimating the region of attraction using a Lyapunov function, the decrease condition
only needs to be met within a bounded sublevel set ``\\{ x : V(x) ≤ ρ \\}``.
The returned [`RoAAwareDecreaseCondition`](@ref) enforces the decrease condition represented
by `cond` only in that sublevel set.

# Arguments
  - `cond::AbstractLyapunovDecreaseCondition`: specifies the loss to be applied when
  ``V(x) ≤ ρ``.

# Keyword Arguments
  - `ρ`: the target level such that the RoA will be ``\\{ x : V(x) ≤ ρ \\}``, defaults to 1.
  - `out_of_RoA_penalty::Function`: specifies the loss to be applied when ``V(x) > ρ``,
    defaults to no loss.
  - `sigmoid::Function`: approximately one when the input is positive and approximately zero
    when the input is negative, defaults to unit step function.

The loss applied to samples ``x`` such that ``V(x) > ρ`` is
``\\lvert \\texttt{out\\_of\\_RoA\\_penalty}(V(x), V̇(x), x, x_0, ρ) \\rvert^2``.

The `sigmoid` function allows for a smooth transition between the ``V(x) ≤ ρ`` case and the
``V(x) > ρ`` case, by combining the above equations into one:

``\\texttt{sigmoid}(ρ - V(x)) (\\text{in-RoA expression}) + \\texttt{sigmoid}(V(x) - ρ) (\\text{out-of-RoA expression}) = 0``.

Note that a hard transition, which only enforces the in-RoA equation when ``V(x) ≤ ρ`` and
the out-of-RoA equation when ``V(x) > ρ`` can be provided by a `sigmoid` which is exactly
one when the input is nonnegative and exactly zero when the input is negative.
As such, the default value is `sigmoid(t) = t ≥ zero(t)`.

See also: [`RoAAwareDecreaseCondition`](@ref)
"""
function make_RoA_aware(
        cond::AbstractLyapunovDecreaseCondition;
        ρ = 1.0,
        out_of_RoA_penalty = (V, dVdt, state, fixed_point, _ρ) -> 0.0,
        sigmoid = (x) -> x .≥ zero.(x)
)::RoAAwareDecreaseCondition
    RoAAwareDecreaseCondition(
        cond,
        sigmoid,
        ρ,
        out_of_RoA_penalty
    )
end
