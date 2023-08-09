using ModelingToolkit, LinearAlgebra, DifferentialEquations
using Symbolics: scalarize
using NeuralPDE, Lux
using Optimization, OptimizationOptimisers, OptimizationOptimJL, NLopt
using Plots
using NeuralLyapunov

##################### Define dynamics via ModelingToolkit #####################

@variables t
D = Differential(t)

function Mass(; name, m = 1.0, b = 10.0, xy = [0.0, 0.0], u = [0.0, 0.0])
    ps = @parameters m = m b = b
    sts = @variables pos(t)[1:2]=xy v(t)[1:2]=u
    eqs = scalarize(D.(pos) .~ v)
    ODESystem(eqs, t, [pos..., v...], ps; name)
end

function damping_force(mass)
    -mass.b .* scalarize(mass.v)
end

function Spring(; name, k = 1e4, l = 1.0)
    ps = @parameters k=k l=l
    @variables x(t), dir(t)[1:2]
    ODESystem(Equation[], t, [x, dir...], ps; name)
end

function connect_spring(spring, a, b)
    [spring.x ~ norm(scalarize(a .- b))
        scalarize(spring.dir .~ scalarize(a .- b))]
end

function spring_force(spring)
    -spring.k .* scalarize(spring.dir) .* (spring.x - spring.l) ./ (1e-6 + spring.x)
end

m = 1.0
b = 3.0
xy = [1.0, -1.0]
k = 1e4
l = 1.0
center = [0.0, 0.0]
g = [0.0, -9.81]
@named mass = Mass(m = m, b = b, xy = xy)
@named spring = Spring(k = k, l = l)

eqs = [connect_spring(spring, mass.pos, center)
    scalarize(D.(mass.v) .~ damping_force(mass) / mass.m + spring_force(spring) / mass.m .+ g)]

@named _model = ODESystem(eqs, t, [spring.x; spring.dir; mass.pos], [])
@named model = compose(_model, mass, spring)
sys = structural_simplify(model)

ode_prob = ODEProblem(sys, [], (0.0, 3.0))
sol = solve(ode_prob, Rosenbrock23())
p1 = plot(sol);
p2 = plot(sol, idxs = (mass.pos[1], mass.pos[2]));
p2 = scatter!([center[1]], [center[2]], legend = false)
plot(p1,p2)

################################ Define domain ################################

lb = [-10.0, -10.0, -5.0, -5.0];
ub = [10.0, 10.0, 5.0, 1.0];
fixed_point = [0.0, 0.0, 0.0, m*g[2]/k - l]

####################### Specify neural Lyapunov problem #######################

# Define neural network discretization
dim_state = length(lb)
dim_hidden = 10
dim_output = 3
chain = [
    Lux.Chain(
        Dense(dim_state, dim_hidden, tanh),
        Dense(dim_hidden, dim_hidden, tanh),
        Dense(dim_hidden, 1, use_bias = false),
    ) for _ = 1:dim_output
]

# Define neural network discretization
strategy = QuasiRandomTraining(200)
discretization = PhysicsInformedNN(chain, strategy)

# Define neural Lyapunov structure
structure = NonnegativeNeuralLyapunov(
        dim_output; 
        δ = 1e-6
        )
structure = PositiveSemiDefiniteStructure(dim_output)
minimization_condition = DontCheckNonnegativity(check_fixed_point = false)

# Set up decrease condition
decrease_condition = AsymptoticDecrease(strict = true; check_fixed_point = true)
spec = NeuralLyapunovSpecification(
    structure,
    minimization_condition,
    decrease_condition,
    )

############################# Construct PDESystem #############################

pde_system, network_func = NeuralLyapunovPDESystem(sys, lb, ub, spec; fixed_point = fixed_point)

######################## Construct OptimizationProblem ########################

prob = discretize(pde_system, discretization)
sym_prob = symbolic_discretize(pde_system, discretization)

callback = function (p, l)
    println("loss: ", l)
    return false
end

########################## Solve OptimizationProblem ##########################

res = Optimization.solve(prob, Adam(); callback = callback, maxiters = 300)
prob = Optimization.remake(prob, u0 = res.u);

println("Switching from Adam to BFGS");
res = Optimization.solve(prob, BFGS(); callback = callback, maxiters = 300)

###################### Get numerical numerical functions ######################
V_func, V̇_func, ∇V_func = NumericalNeuralLyapunovFunctions(
    discretization.phi, 
    res.u, 
    network_func, 
    structure.V,
    ODEFunction(sys),
    zeros(4);
    p = p
    )

################################## Simulate ###################################
xs, ys = [lb[i]:0.02:ub[i] for i in eachindex(lb)]
states = Iterators.map(collect, Iterators.product(xs, ys))
V_predict = vec(V_func(hcat(states...)))
dVdt_predict = vec(V̇_func(hcat(states...)))

# Get RoA Estimate
data = reshape(V_predict, (length(xs), length(ys)));
data = vcat(data[1, :], data[end, :], data[:, 1], data[:, end]);
ρ = minimum(data)

# Print statistics
println("V(0.,0.) = ", V_func([0.0, 0.0]))
println("V ∋ [", min(V_func([0.0, 0.0]), minimum(V_predict)), ", ", maximum(V_predict), "]")
println(
    "V̇ ∋ [",
    minimum(dVdt_predict),
    ", ",
    max(V̇_func([0.0, 0.0]), maximum(dVdt_predict)),
    "]",
)

# Plot results

p1 = plot(xs, ys, V_predict, linetype = :contourf, title = "V", xlabel = "x", ylabel = "ẋ");
p1 = scatter!([0], [0], label = "Equilibrium");
p2 = plot(
    xs,
    ys,
    dVdt_predict,
    linetype = :contourf,
    title = "dV/dt",
    xlabel = "x",
    ylabel = "ẋ",
);
p2 = scatter!([0], [0], label = "Equilibrium");
p3 = plot(
    xs,
    ys,
    V_predict .< ρ,
    linetype = :contourf,
    title = "Estimated RoA",
    xlabel = "x",
    ylabel = "ẋ",
    colorbar = false,
);
p4 = plot(
    xs,
    ys,
    dVdt_predict .< 0,
    linetype = :contourf,
    title = "dV/dt < 0",
    xlabel = "x",
    ylabel = "ẋ",
    colorbar = false,
);
p4 = scatter!([0], [0], label = "Equilibrium");
plot(p1, p2, p3, p4)
