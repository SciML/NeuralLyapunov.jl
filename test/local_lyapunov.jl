using NeuralLyapunov, CSDP, ForwardDiff, Test, LinearAlgebra

################################### ẋ = -x ###################################
# Set up dynamics with a known Lyapunov function: V(x) = C x ⋅ x
f1(x, p, t) = -x

# Set up Jacobian
J1(x, p) = (-1.0*I)(3)

# Find Lyapunov function and rescale for C = 1
_V1, _V̇1 = local_lyapunov(f1, 3, CSDP.Optimizer, J1)
V1(x) = _V1(x) ./ _V1([0, 0, 1])

# Test equality
xs = -1:0.1:1
states = Iterators.map(collect, Iterators.product(xs, xs, xs))
errs1 = @. abs(V1(states) - states ⋅ states)
@test all(errs1 .< 1e-10)

########################## Simple harmonic oscillator #########################
function f2(state, p, t)
    ζ, ω_0 = p
    pos, vel = state
    vcat(vel, -2ζ * vel - ω_0^2 * pos)
end
p2 = [3.2, 5.1]

# Find Lyapunov function and derivative
V2, V̇2 = local_lyapunov(f2, 2, CSDP.Optimizer; p = p2)
dV2dt = (state) -> ForwardDiff.derivative((δt) -> V2(state + δt * f2(state, p2, 0.0)), 0.0)

# Test V̇2 = d/dt V2
xs = -1:0.03:1
states = Iterators.map(collect, Iterators.product(xs, xs))
errs2 = @. abs(dV2dt(states) - V̇2(states))
@test all(errs2 .< 1e-10)

# Test V̇2 ≺ 0
@test all( @. V̇2(states) < 0)
@test V̇2(zeros(2)) == 0.0
