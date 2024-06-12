using NeuralLyapunov, CSDP, ForwardDiff, Test, LinearAlgebra

################################### ẋ = -x ###################################
# Set up dynamics with a known Lyapunov function: V(x) = C x ⋅ x
f1(x, p, t) = -x

# Set up Jacobian
J1(x, p) = (-1.0 * I)(3)

# Find Lyapunov function and rescale for C = 1
_V1, _V̇1 = local_lyapunov(f1, 3, CSDP.Optimizer, J1)
V1(x) = _V1(x) ./ _V1([0, 0, 1])

# Test equality
xs1 = -1:0.1:1
states1 = Iterators.map(collect, Iterators.product(xs1, xs1, xs1))
errs1 = @. abs(V1(states1) - states1 ⋅ states1)
@test all(errs1 .< 1e-10)

########################## Simple harmonic oscillator #########################
function f2(state::AbstractVector, p, t)
    ζ, ω_0 = p
    pos, vel = state
    vcat(vel, -2ζ * ω_0 * vel - ω_0^2 * pos)
end
function f2(states::AbstractMatrix, p, t)
    ζ, ω_0 = p
    pos, vel = states[1, :], states[2, :]
    vcat(transpose(vel), transpose(-2ζ * ω_0 * vel - ω_0^2 * pos))
end
p2 = [3.2, 5.1]

# Find Lyapunov function and derivative
V2, V̇2 = local_lyapunov(f2, 2, CSDP.Optimizer; p = p2)
dV2dt = (state) -> ForwardDiff.derivative((δt) -> V2(state + δt * f2(state, p2, 0.0)), 0.0)

# Test V̇2 = d/dt V2
xs2 = -1:0.03:1
states2 = hcat(Iterators.map(collect, Iterators.product(xs2, xs2))...)
errs2 = abs.(V̇2(states2) .- dV2dt(states2))
@test all(errs2 .< 1e-10)

# Test V̇2 ≺ 0
@test all(V̇2(states2) .< 0)
@test V̇2(zeros(2)) == 0.0
