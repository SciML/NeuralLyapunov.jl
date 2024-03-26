using LinearAlgebra
using NeuralPDE, Lux, ModelingToolkit
using Optimization, OptimizationOptimisers, OptimizationOptimJL, NLopt
using NeuralLyapunov
using Random
using Test

Random.seed!(200)

println("Planar Quadrotor - Policy Search")

######################### Define dynamics and domain ##########################

@parameters m I r g
defaults = Dict([m => 1.0, I => 1.0, r => 1.0, g => 1.0])

@variables t x(t) y(t) θ(t) u1(t) [input = true] u2(t) [input = true]
Dt = Differential(t)
DDt = Dt^2

eqs = [
    m * DDt(x) ~ -(-u1 + u2) * sin(θ),
    m * DDt(y) ~ (u1 + u2) * cos(θ) - m * g,
    I * DDt(θ) ~ r * (u1 - u2)
]

@named planar_quadrotor = ODESystem(
    eqs,
    t,
    [x, y, θ, u1, u2],
    [m, I, r, g];
    defaults = defaults
)

(planar_quadrotor_open_loop_dynamics, _), state_order, p_order = ModelingToolkit.generate_control_function(
    planar_quadrotor; simplify = true)

bounds = [
    θ ∈ (-π, π),
    x ∈ (-10.0, 10.0),
    y ∈ (-10.0, 10.0),
    Dt(θ) ∈ (-10.0, 10.0),
    Dt(x) ∈ (-10.0, 10.0),
    Dt(y) ∈ (-10.0, 10.0)
]
equilibrium = zeros(length(bounds));
p = [defaults[param] for param in p_order]

####################### Specify neural Lyapunov problem #######################

# Define neural network discretization
# We use an input layer that is periodic with period 2π with respect to θ
dim_state = length(bounds)
dim_hidden = 20
dim_phi = 5
dim_u = 2
dim_output = dim_phi + dim_u
chain = [Lux.Chain(
             Lux.WrappedFunction(x -> vcat(
                 x[1:(end - 1), :],
                 transpose(sin.(x[end, :])),
                 transpose(cos.(x[end, :]))
             )),
             Dense(7, dim_hidden, tanh),
             Dense(dim_hidden, dim_hidden, tanh),
             Dense(dim_hidden, dim_hidden, tanh),
             Dense(dim_hidden, 1)
         ) for _ in 1:dim_output]

# Define neural network discretization
strategy = QuasiRandomTraining(1000)
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
decrease_condition = make_RoA_aware(
    LyapunovDecreaseCondition(
        true,
        (V, dVdt) -> dVdt,
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
    planar_quadrotor,
    bounds,
    spec
)

######################## Construct OptimizationProblem ########################

sym_prob = symbolic_discretize(pde_system, discretization)
prob = discretize(pde_system, discretization)

########################## Solve OptimizationProblem ##########################

res = Optimization.solve(prob, OptimizationOptimisers.Adam(0.01); maxiters = 500)
prob = Optimization.remake(prob, u0 = res.u)
res = Optimization.solve(prob, OptimizationOptimisers.Adam(); maxiters = 500)
prob = Optimization.remake(prob, u0 = res.u)
res = Optimization.solve(prob, BFGS(); maxiters = 300)

###################### Get numerical numerical functions ######################

V_func, V̇_func, ∇V_func = NumericalNeuralLyapunovFunctions(
    discretization.phi,
    res.u,
    network_func,
    structure,
    planar_quadrotor_open_loop_dynamics,
    equilibrium;
    p = p
)

controller = get_policy(discretization.phi, res.u, network_func, dim_u)

################################## Simulate ###################################

lb = [-10.0, -10.0, -10.0, -10.0, -10.0, -pi];
ub = [10.0, 10.0, 10.0, 10.0, 10.0, pi];

#=
b = [l:0.02:u for (l,u) in zip(lb, ub)]
states = Iterators.map(collect, Iterators.product(b...))
V_predict = vec(V_func(hcat(states...)))
dVdt_predict = vec(V̇_func(hcat(states...)))
=#

#################################### Tests ####################################

# Training should result in a fixed point at the upright equilibrium
@test all(isapprox.(
    planar_quadrotor_open_loop_dynamics(equilibrium, controller(equilibrium), p, 0.0), 0.0;
    atol = 1e-8))

# V̇ should be negative almost everywhere
#@test sum(dVdt_predict .> 0) / length(dVdt_predict) < 1e-3

################################## Simulate ###################################

using DifferentialEquations

state_order = map(st -> istree(st) ? operation(st) : st, state_order)
state_syms = Symbol.(state_order)

closed_loop_dynamics = ODEFunction(
    (x, p, t) -> planar_quadrotor_open_loop_dynamics(x, controller(x), p, t);
    sys = SciMLBase.SymbolCache(state_syms, Symbol.(p_order))
)

# Starting still and too high
x0 = [0.0, 0.0, 0.0, 0.0, 0.1, 0.0]
ode_prob = ODEProblem(closed_loop_dynamics, x0, [0.0, 10.0], p)
sol = solve(ode_prob)
p1 = plot(sol[:x], sol[:y], label = "Trajectory");
p1 = scatter!([sol[:x][1]], [sol[:y][1]], label = "Start");
p1 = scatter!([0], [0], label = "Goal");
p1 = scatter!([sol[:x][end]], [sol[:y][end]], label = "End")

# Should make it to the top
θ_end, ω_end = sol.u[end]
x_end, y_end = sin(θ_end), -cos(θ_end)
@test all(isapprox.([x_end, y_end, ω_end], [0.0, 1.0, 0.0]; atol = 1e-3))

# Starting at a random point
x0 = lb .+ rand(length(bounds)) .* (ub .- lb)
ode_prob = ODEProblem(closed_loop_dynamics, x0, [0.0, 20.0], p)
sol = solve(ode_prob, Tsit5())
# plot(sol)

# Should make it to the top
θ_end, ω_end = sol.u[end]
x_end, y_end = sin(θ_end), -cos(θ_end)
@test all(isapprox.([x_end, y_end, ω_end], [0.0, 1.0, 0.0]; atol = 1e-3))

#=
# Print statistics
println("V(π, 0) = ", V_func(upright_equilibrium))
println(
    "f([π, 0], u([π, 0])) = ",
    controlled_pendulum_dynamics(upright_equilibrium, u(upright_equilibrium), p, 0.0)
)
println(
    "V ∋ [",
    min(V_func(upright_equilibrium),
    minimum(V_predict)),
    ", ",
    maximum(V_predict),
    "]"
)
println(
    "V̇ ∋ [",
    minimum(dVdt_predict),
    ", ",
    max(V̇_func(upright_equilibrium), maximum(dVdt_predict)),
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
