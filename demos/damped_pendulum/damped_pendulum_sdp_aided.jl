using LinearAlgebra
using NeuralPDE, Lux, ModelingToolkit
import Hypatia, JuMP
using ForwardDiff
using Optimization, OptimizationOptimisers, OptimizationOptimJL, NLopt
using Plots
using NeuralLyapunov

# Define dynamics
"Pendulum Dynamics"
function pendulum_dynamics(state::AbstractMatrix{T})::AbstractMatrix{T} where T <:Number
    pos = transpose(state[1,:]); vel = transpose(state[2,:])
    vcat(vel, -vel-sin.(pos))
end
function pendulum_dynamics(state::AbstractVector{T})::AbstractVector{T} where T <:Number
    pos = state[1]; vel = state[2]
    vcat(vel, -vel-sin.(pos))
end
lb = [0.0, -10.0]; ub = [4*pi, 10.0]
fixed_point = [2*pi, 0.0]

# Make log version
dim_output = 2
κ=20.0
δ=0.1
pde_system_log, lyapunov_func = NeuralLyapunovPDESystem(pendulum_dynamics, lb, ub, dim_output, δ=δ, relu=(t)->log(1.0 + exp( κ * t))/κ, fixed_point=fixed_point)

# Define neural network discretization
dim_state = length(lb)
dim_hidden = 15
chain = [Lux.Chain(
                Dense(dim_state, dim_hidden, tanh), 
                Dense(dim_hidden, dim_hidden, tanh),
                Dense(dim_hidden, 1, use_bias=false)
                )
            for _ in 1:dim_output
            ]

# Define neural network discretization
# strategy = GridTraining(0.1)
# strategy = QuasiRandomTraining(100)
strategy = QuadratureTraining()
discretization = PhysicsInformedNN(chain, strategy)

# Build optimization problem
prob_log = discretize(pde_system_log, discretization)
sym_prob_log = symbolic_discretize(pde_system_log, discretization)

callback = function (p, l)
    println("loss: ", l)
    return false
end

# Optimize with stricter log version
res = Optimization.solve(prob_log, Adam(); callback=callback, maxiters=300)

# Rebuild with weaker ReLU version
pde_system_relu, _ = NeuralLyapunovPDESystem(pendulum_dynamics, lb, ub, dim_output, δ=δ, fixed_point=fixed_point)
prob_relu = discretize(pde_system_relu, discretization)
sym_prob_relu = symbolic_discretize(pde_system_relu, discretization)
prob_relu = Optimization.remake(prob_relu, u0=res.u); println("Switching from log(1 + κ exp(V̇))/κ to max(0,V̇)")
res = Optimization.solve(prob_relu, Adam(); callback=callback, maxiters=300)
prob_relu = Optimization.remake(prob_relu, u0=res.u); println("Switching from Adam to BFGS")
res = Optimization.solve(prob_relu, BFGS(); callback=callback, maxiters=300)

# Get numerical numerical functions
V_func, V̇_func = NumericalNeuralLyapunovFunctions(discretization.phi, res, lyapunov_func, pendulum_dynamics)

# Simulate
xs,ys = [lb[i]:0.02:ub[i] for i in eachindex(lb)]
states = Iterators.map(collect, Iterators.product(xs, ys))
V_predict = vec(V_func(hcat(states...)))
dVdt_predict = vec(V̇_func(hcat(states...)))
# V_predict = [V_func([x0,y0]) for y0 in ys for x0 in xs]
# dVdt_predict  = [V̇_func([x0,y0]) for y0 in ys for x0 in xs]

##################### Get SDP RoA Estimate ##########################
# Calculate quadratic Lyapunov function of linearized system with SDP
A = ForwardDiff.jacobian(pendulum_dynamics, fixed_point)
model = JuMP.Model(Hypatia.Optimizer)
JuMP.set_silent(model)
JuMP.@variable(model, P[1:2, 1:2], PSD)
JuMP.@variable(model, Q[1:2, 1:2], PSD)
JuMP.@constraint(model, P*A+transpose(A)*P .== -(Q))
JuMP.optimize!(model)
Psol = JuMP.value.(P) 

# Numerical form of Lyapunov function
V_func_SDP(state::AbstractVector) = dot(state - fixed_point, Psol, state - fixed_point)
V_func_SDP(states::AbstractMatrix) = mapslices(V_func_SDP, states, dims=[1])

# Numerical gradient of Lyapunov function
∇V_func_SDP(state::AbstractVector) = ForwardDiff.gradient(V_func_SDP, state)
∇V_func_SDP(states::AbstractMatrix) = mapslices(∇V_func_SDP, states, dims=[1])

# Numerical time derivative of Lyapunov function
V̇_func_SDP(state::AbstractVector) = pendulum_dynamics(state) ⋅ ∇V_func_SDP(state)
V̇_func_SDP(states::AbstractMatrix) = reshape(map(x->x[1]⋅x[2], zip(eachslice(pendulum_dynamics(states), dims=2), eachslice(∇V_func_SDP(states), dims=2))), (1,:))

# Simulate
V_predict_SDP = vec(V_func_SDP(hcat(states...)))
dVdt_predict_SDP = vec(V̇_func_SDP(hcat(states...)))

# Get RoA Estimate
data = reshape(V_predict_SDP, (length(xs), length(ys)));
edges = vcat(data[1,:], data[end,:], data[:,1],data[:,end]);
ρ_max_SDP = minimum(edges)
ρ_min_SDP = 0.0
ρ_SDP = ρ_max_SDP
while true # abs(ρ_max - ρ_min) > maximum(data)*1e-6
    marginal_RoA_est = ρ_min_SDP .< V_predict_SDP .< ρ_SDP;
    if sum(marginal_RoA_est) == 0
        global ρ_SDP = ρ_min_SDP
        break
    end
    if maximum(dVdt_predict_SDP[marginal_RoA_est]) > 0
        global ρ_max_SDP = ρ_SDP
    else
        global ρ_min_SDP = ρ_SDP
    end
    global ρ_SDP = (ρ_max_SDP +ρ_min_SDP)/2
end

RoA_est_SDP = V_predict_SDP .< ρ_SDP;

################## Get Neural Lyapunov RoA Estimate ###################

# Get RoA Estimate
data = reshape(V_predict, (length(xs), length(ys)));
egdes = vcat(data[1,:], data[end,:], data[:,1],data[:,end]);
ρ_max = minimum(edges)
ρ_min = maximum(V_predict[RoA_est_SDP .&& V_predict .< minimum(V_predict[.!RoA_est_SDP])])
ρ = ρ_max
while true # abs(ρ_max - ρ_min) > maximum(data)*1e-6
    marginal_RoA_est = (ρ_min .< V_predict .< ρ) # .&& .! RoA_est_SDP; # not sure if I can include this
    if sum(marginal_RoA_est) == 0
        global ρ = ρ_min
        break
    end
    if maximum(dVdt_predict[marginal_RoA_est]) > 0
        global ρ_max = ρ
    else
        global ρ_min = ρ
    end
    global ρ = (ρ_max +ρ_min)/2
end

# Print statistics
println("V(2π, 0) = ", V_func(fixed_point))
println("V ∋ [", min(V_func(fixed_point), minimum(V_predict)), ", ", maximum(V_predict), "]")
println("V̇ ∋ [", minimum(dVdt_predict), ", ", max(V̇_func(fixed_point), maximum(dVdt_predict)), "]")
println("Certified V ∈ [0.0, ", ρ, ")")

# Plot results

p1 = plot(xs, ys, V_predict, linetype=:contourf, title = "V", xlabel="x", ylabel="ẋ");
p2 = plot(xs, ys, dVdt_predict, linetype=:contourf, title="dV/dt", xlabel="x", ylabel="ẋ");
p2 = scatter!((lb[1]+pi):2*pi:ub[1], zeros(4), label="Unstable equilibria", markershape=:x);
p2 = scatter!(lb[1]:2*pi:ub[1], zeros(5), label="Stable equilibria", markershape=:+);
p3 = plot(xs, ys, V_predict .≤ ρ, linetype=:contourf, title="Estimated RoA", xlabel="x", ylabel="ẋ", colorbar=false);
p3 = scatter!((lb[1]+pi):2*pi:ub[1], zeros(4), label="Unstable equilibria", markershape=:x);
p3 = scatter!(lb[1]:2*pi:ub[1], zeros(5), label="Stable equilibria", markershape=:+);
p4 = plot(xs,ys, dVdt_predict.<0, linetype=:contourf, title="dV/dt<0", xlabel="x", ylabel="ẋ", colorbar=false);
p4 = scatter!((lb[1]+pi):2*pi:ub[1], zeros(4), label="Unstable equilibria", markershape=:x);
p4 = scatter!(lb[1]:2*pi:ub[1], zeros(5), label="Stable equilibria", markershape=:+);
plot(p1, p2, p3, p4)
# savefig("Pendulum")

#=
p1 = plot(xs, ys, V_predict_SDP, linetype=:contourf, title = "V", xlabel="x", ylabel="ẋ");
p2 = plot(xs, ys, dVdt_predict_SDP, linetype=:contourf, title="dV/dt", xlabel="x", ylabel="ẋ");
p2 = scatter!((lb[1]+pi):2*pi:ub[1], zeros(4), label="Unstable equilibria");
p2 = scatter!(lb[1]:2*pi:ub[1], zeros(5), label="Stable equilibria");
p3 = plot(xs, ys, V_predict_SDP .≤ ρ_SDP, linetype=:contourf, title="Estimated RoA", xlabel="x", ylabel="ẋ", colorbar=false);
p3 = scatter!((lb[1]+pi):2*pi:ub[1], zeros(4), label="Unstable equilibria");
p3 = scatter!(lb[1]:2*pi:ub[1], zeros(5), label="Stable equilibria");
p4 = plot(xs,ys, dVdt_predict_SDP.<0, linetype=:contourf, title="dV/dt<0", xlabel="x", ylabel="ẋ", colorbar=false);
p4 = scatter!((lb[1]+pi):2*pi:ub[1], zeros(4), label="Unstable equilibria");
p4 = scatter!(lb[1]:2*pi:ub[1], zeros(5), label="Stable equilibria");
plot(p1, p2, p3, p4)
=#
