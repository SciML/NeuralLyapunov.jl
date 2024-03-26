using LinearAlgebra
using NeuralPDE, Lux, ModelingToolkit
using Optimization, OptimizationOptimisers, OptimizationOptimJL, NLopt
using DifferentialEquations
using NeuralLyapunov
using Random
using Test

Random.seed!(200)

println("Planar Car - Policy Search")

######################### Define dynamics and domain ##########################

function open_loop_car_dynamics(state, u, p, t)
    x, y, θ = state
    v, ω = u
    return [
        v * cos(θ)
        v * sin(θ)
        ω
    ]
end

lb = [-1.0, -1.0, -π];
ub = [1.0, 1.0, π];
goal = [0.0, 0.0, 0.0]
state_syms = [:x, :y, :θ]

####################### Specify neural Lyapunov problem #######################

# Define neural network discretization
# We use an input layer that is periodic with period 2π with respect to θ
dim_state = length(state_syms)
dim_hidden = 15
dim_phi = 1
dim_u = 2
dim_output = dim_phi + dim_u
chain = [Chain(
             WrappedFunction(x -> vcat(
                 x[1:end-1, :],
                 transpose(sin.(x[end, :])),
                 transpose(cos.(x[end, :])),
             )),
             Dense(dim_state + 1, dim_hidden, tanh),
             Dense(dim_hidden, dim_hidden, tanh),
             Dense(dim_hidden, dim_hidden, tanh),
             Dense(dim_hidden, 1)
         ) for _ in 1:dim_output]

# Define neural network discretization
strategy = GridTraining(0.1)
# strategy = QuasiRandomTraining(1000)
discretization = PhysicsInformedNN(chain, strategy)

# Define neural Lyapunov structure
structure = PositiveSemiDefiniteStructure(
    dim_phi;
    pos_def = function (state, fixed_point)
        θ = state[end]
        st = state[1:(end - 1)]
        θ_eq = fixed_point[end]
        fp = fixed_point[1:(end - 1)]
        log(1.0 + (sin(θ) - sin(θ_eq))^2 + (cos(θ) - cos(θ_eq))^2 + (st - fp) ⋅ (st - fp))
    end
)
structure = add_policy_search(
    structure,
    dim_u
)
minimization_condition = DontCheckNonnegativity()

# Define Lyapunov decrease condition
# decrease_condition = AsymptoticDecrease(strict = true)
decrease_condition = make_RoA_aware(
    LyapunovDecreaseCondition(
        true,
        (V, dVdt) -> dVdt + 10.0 * V,
        function (st, fp)
            _st = vcat(st[1:(end - 1)], sin(st[end]), cos(st[end]))
            _fp = vcat(fp[1:(end - 1)], sin(fp[end]), cos(fp[end]))
            return -1e-5 * (_st - _fp) ⋅ (_st - _fp)
        end,
        (t) -> max(0.0, t)
    )
)
# Construct neural Lyapunov specification
spec = NeuralLyapunovSpecification(
    structure,
    minimization_condition,
    decrease_condition
)

############################# Construct PDESystem #############################

pde_system, network_func = NeuralLyapunovPDESystem(
    open_loop_car_dynamics,
    lb,
    ub,
    spec;
    fixed_point = goal,
    state_syms = state_syms,
    policy_search = true
)

######################## Construct OptimizationProblem ########################

sym_prob = symbolic_discretize(pde_system, discretization);
prob = discretize(pde_system, discretization);

########################## Solve OptimizationProblem ##########################

res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.01); maxiters = 300)
prob = Optimization.remake(prob, u0 = res.u)
res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.001); maxiters = 300)
prob = Optimization.remake(prob, u0 = res.u)
res = Optimization.solve(prob, BFGS(); maxiters = 500)

###################### Get numerical numerical functions ######################

V_func, V̇_func = NumericalNeuralLyapunovFunctions(
    discretization.phi,
    res.u,
    network_func,
    structure,
    open_loop_car_dynamics,
    goal
)

controller = get_policy(discretization.phi, res.u, network_func, dim_u)

################################## Simulate ###################################

bounds = [l:0.1:u for (l,u) in zip(lb, ub)]
states = Iterators.map(collect, Iterators.product(bounds...))
V_predict = vec(V_func(hcat(states...)))
dVdt_predict = vec(V̇_func(hcat(states...)))

#################################### Tests ####################################

# Training should result in a fixed point at the upright equilibrium
@test all(isapprox.(
    open_loop_car_dynamics(goal, controller(goal), SciMLBase.NullParameters(), 0.0),
    0.0; atol = 1e-8))

# V̇ should be negative almost everywhere
@test sum(dVdt_predict .> 0) / length(dVdt_predict) < 1e-3

################################## Simulate ###################################

using Plots

function plot_planar_car(sol; fixed_point = goal)
    x = sol[:x]
    y = sol[:y]
    θ = sol[:θ]
    v = first.(controller.(sol.u))
    vx = v .* cos.(θ)
    vy = v .* sin.(θ)
    Δt = diff(sol.t)
    push!(Δt, Δt[end])
    p = quiver(x, y, quiver = (vx .* Δt, vy .* Δt))
    p = scatter!([x[1]], [y[1]], marker=:o, label="Initial state")
    p = scatter!([fixed_point[1]], [fixed_point[2]], marker=:x, label="Goal")
    return p
end


closed_loop_dynamics = ODEFunction(
    (x, p, t) -> open_loop_car_dynamics(x, controller(x), p, t);
    sys = SciMLBase.SymbolCache(state_syms)
)

# Starting at goal
ode_prob = ODEProblem(closed_loop_dynamics, goal, [0.0, 1.0])
sol = solve(ode_prob, Tsit5())
# plot(sol)

# Should make it to the goal
x_end, y_end, θ_end = sol.u[end]
s_end, c_end = sin(θ_end), cos(θ_end)
s_goal, c_goal = sin(goal[end]), cos(goal[end])
@test all(isapprox.([x_end, y_end, s_end, c_end], vcat(goal[1:end-1], s_goal, c_goal); atol = 1e-5))

# Starting in the right direction
x0 = [-0.2, 0.0, 0.0]
@show V_func(x0);
@assert V_func(x0) < 1
ode_prob = ODEProblem(closed_loop_dynamics, x0, [0.0, 0.3])
sol = solve(ode_prob, Tsit5())
plot(sol)
plot_planar_car(sol)

# Should make it to the goal
x_end, y_end, θ_end = sol.u[end]
s_end, c_end = sin(θ_end), cos(θ_end)
@test all(isapprox.([x_end, y_end, s_end, c_end], vcat(goal[1:end-1], s_goal, c_goal); atol = 1e-3))

# Starting in the wrong direction
x0 = [0.0, 0.0, π/4]
@show V_func(x0);
@assert V_func(x0) < 1
ode_prob = ODEProblem(closed_loop_dynamics, x0, [0.0, 0.3])
sol = solve(ode_prob, Tsit5())
plot(sol)
plot_planar_car(sol)

# Should make it to the goal
x_end, y_end, θ_end = sol.u[end]
s_end, c_end = sin(θ_end), cos(θ_end)
@test all(isapprox.([x_end, y_end, s_end, c_end], vcat(goal[1:end-1], s_goal, c_goal); atol = 0.01))


#=
# Print statistics
println("V(π, 0) = ", V_func(goal))
println(
    "f([π, 0], u([π, 0])) = ",
    open_loop_pendulum_dynamics(goal, u(goal), p, 0.0)
)
println(
    "V ∋ [",
    min(V_func(goal),
    minimum(V_predict)),
    ", ",
    maximum(V_predict),
    "]"
)
println(
    "V̇ ∋ [",
    minimum(dVdt_predict),
    ", ",
    max(V̇_func(goal), maximum(dVdt_predict)),
    "]"
)

# Plot results
using Plots

p1 = plot(
    xs / pi,
    ys,
    V_predict,
    linetype =
    :contourf,
    title = "V",
    xlabel = "θ/π",
    ylabel = "ω",
    c = :bone_1
);
p1 = scatter!([-2 * pi, 0, 2 * pi] / pi, [0, 0, 0],
    label = "Downward Equilibria", color = :green, markershape = :+);
p1 = scatter!(
    [-pi, pi] / pi, [0, 0], label = "Upward Equilibria", color = :red, markershape = :x);
p2 = plot(
    xs / pi,
    ys,
    dVdt_predict,
    linetype = :contourf,
    title = "dV/dt",
    xlabel = "θ/π",
    ylabel = "ω",
    c = :binary
);
p2 = scatter!([-2 * pi, 0, 2 * pi] / pi, [0, 0, 0],
    label = "Downward Equilibria", color = :green, markershape = :+);
p2 = scatter!([-pi, pi] / pi, [0, 0], label = "Upward Equilibria",
    color = :red, markershape = :x, legend = false);
p3 = plot(
    xs / pi,
    ys,
    dVdt_predict .< 0,
    linetype = :contourf,
    title = "dV/dt < 0",
    xlabel = "θ/π",
    ylabel = "ω",
    colorbar = false,
    linewidth = 0
);
p3 = scatter!([-2 * pi, 0, 2 * pi] / pi, [0, 0, 0],
    label = "Downward Equilibria", color = :green, markershape = :+);
p3 = scatter!([-pi, pi] / pi, [0, 0], label = "Upward Equilibria",
    color = :red, markershape = :x, legend = false);
plot(p1, p2, p3)
=#
