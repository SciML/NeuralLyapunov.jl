"""
    RoAAwareDecreaseCondition(check_decrease, decrease, strength, relu, check_fixed_point,
                              ρ, out_of_RoA_penalty)

Specifies the form of the Lyapunov conditions to be used, training for a region of
attraction estimate of `{ x : V(x) ≤ ρ }`

If `check_decrease`, training will enforce
`decrease(V(state), dVdt(state)) ≤ strength(state, fixed_point)` whenever `V(state) ≤ ρ`,
and will instead apply
`|out_of_RoA_penalty(V(state), dVdt(state), state, fixed_point, ρ)|^2` when `V(state) > ρ`.

The inequality will be approximated by the equation
    `relu(decrease(V, dVdt) - strength(state, fixed_point)) = 0.0`.
If `check_fixed_point` is `false`, then training assumes `dVdt(fixed_point) = 0`, but
if `check_fixed_point` is `true`, then training will attempt to enforce
`dVdt(fixed_point) = 0`.

If the dynamics truly have a fixed point at `fixed_point` and `dVdt` has been defined
properly in terms of the dynamics, then `dVdt(fixed_point)` will be `0` and there is no need
to enforce `dVdt(fixed_point) = 0`, so `check_fixed_point` defaults to `false`.

# Examples:

Asymptotic decrease can be enforced by requiring
    `dVdt ≤ -C |state - fixed_point|^2`,
which corresponds to
    `decrease = (V, dVdt) -> dVdt` and
    `strength = (x, x0) -> -C * (x - x0) ⋅ (x - x0)`.

Exponential decrease of rate `k` is proven by `dVdt ≤ - k * V`, so corresponds to
    `decrease = (V, dVdt) -> dVdt + k * V` and
    `strength = (x, x0) -> 0.0`.

Enforcing either condition only in the region of attraction and not penalizing any points
outside that region would correspond to
    `out_of_RoA_penalty = (V, dVdt, state, fixed_point, ρ) -> 0.0`,
whereas an example of a penalty that decays farther in state space from the fixed point is
    `out_of_RoA_penalty = (V, dVdt, state, fixed_point, ρ) -> 1.0 / ((x - x0) ⋅ (x - x0))`.
Note that this penalty could also depend on values of `V`, `dVdt`, and `ρ`.
"""
struct RoAAwareDecreaseCondition <: AbstractLyapunovDecreaseCondition
    check_decrease::Bool
    decrease::Function
    strength::Function
    relu::Function
    check_fixed_point::Bool
    ρ::Real
    out_of_RoA_penalty::Function
end

function check_decrease(cond::RoAAwareDecreaseCondition)::Bool
    cond.check_decrease
end

function check_stationary_fixed_point(cond::RoAAwareDecreaseCondition)::Bool
    cond.check_fixed_point
end

function get_decrease_condition(cond::RoAAwareDecreaseCondition)
    if cond.check_decrease
        return  function (V, dVdt, x, fixed_point)
                    (V(x) ≤ cond.ρ) * cond.relu(
                        cond.decrease(V(x), dVdt(x)) - cond.strength(x, fixed_point)
                    ) +
                    (V(x) > cond.ρ) * cond.out_of_RoA_penalty(V(x), dVdt(x), x, fixed_point,
                                                              cond.ρ)
                end
    else
        return nothing
    end
end

"""
    make_RoA_aware(cond; ρ, out_of_RoA_penalty)

Adds awareness of the region of attraction (RoA) estimation task to the supplied
`LyapunovDecreaseCondition`

`ρ` is the target level such that the RoA will be `{ x : V(x) ≤ ρ }`.
`cond` specifies the loss applied when `V(state) ≤ ρ`, and
`|out_of_RoA_penalty(V(state), dVdt(state), state, fixed_point, ρ)|^2` is the loss from
`state` values such that `V(state) > ρ`.
"""
function make_RoA_aware(
        cond::LyapunovDecreaseCondition;
        ρ = 1.0,
        out_of_RoA_penalty = (V, dVdt, state, fixed_point, _ρ) -> 0.0
)::RoAAwareDecreaseCondition
    RoAAwareDecreaseCondition(
        cond.check_decrease,
        cond.decrease,
        cond.strength,
        cond.relu,
        cond.check_fixed_point,
        ρ,
        out_of_RoA_penalty
    )
end
