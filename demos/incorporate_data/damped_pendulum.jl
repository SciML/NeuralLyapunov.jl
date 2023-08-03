using LinearAlgebra
using NeuralPDE, Lux, ModelingToolkit
using Optimization, OptimizationOptimisers, OptimizationOptimJL, NLopt
using Plots
using NeuralLyapunov
using Random

Random.seed!(200)

######################### Define dynamics and domain ##########################

function f_true(x, p, t)
    θ, ω = x
    ζ, ω_0 = p
    [ω;
    -2ζ * ω - ω_0^2 * sin(θ)]
end
lb = [-pi, -10.0];
ub = [pi, 10.0];
p = [0.5, 1.0]
true_dynamics = ODEFunction(f_true; syms = [:θ, :ω], paramsyms = [:ζ, :ω_0])

####################### Specify neural Lyapunov problem #######################

# Define neural network discretization
# We use an input layer that is periodic with period 2π with respect to θ
dim_state = length(lb)
dim_hidden = 15
dim_output = 2
chain = [
    Lux.Chain(
        Lux.WrappedFunction(x -> vcat(
            transpose(sin.(x[1,:])), 
            transpose(cos.(x[1,:])), 
            transpose(x[2,:])
            )),
        Dense(3, dim_hidden, tanh),
        Dense(dim_hidden, dim_hidden, tanh),
        Dense(dim_hidden, 1, use_bias = false),
    ) for _ = 1:dim_output
]

# Define neural network discretization
strategy = GridTraining(0.1)
discretization = PhysicsInformedNN(chain, strategy)

# Define neural Lyapunov structure
structure = NonnegativeNeuralLyapunov(
        dim_output; 
        δ = 1e-6
        )
minimization_condition = DontCheckNonnegativity(check_fixed_point = true)

# Define Lyapunov decrease condition
decrease_condition = AsymptoticDecrease(strict = true)

# Construct neural Lyapunov specification
spec = NeuralLyapunovSpecification(
    structure,
    minimization_condition,
    decrease_condition,
    )

############################# Construct PDESystem #############################

pde_system, network_func = NeuralLyapunovPDESystem(
    true_dynamics,
    lb,
    ub,
    spec;
    p = p
)

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
res = Optimization.solve(prob, Adam(); callback = callback, maxiters = 300)

println("Switching from Adam to BFGS");
prob = Optimization.remake(prob, u0 = res.u);
res = Optimization.solve(prob, BFGS(); callback = callback, maxiters = 300)

###################### Get numerical numerical functions ######################
V_func, V̇_func, ∇V_func = NumericalNeuralLyapunovFunctions(
    discretization.phi, 
    res, 
    network_func, 
    structure.V,
    true_dynamics,
    zeros(length(lb));
    p = p
    )

################################## Simulate ###################################
xs = 2*lb[1]:0.02:2*ub[1]
ys = lb[2]:0.02:ub[2]
states = Iterators.map(collect, Iterators.product(xs, ys))
V_predict = vec(V_func(hcat(states...)))
dVdt_predict = vec(V̇_func(hcat(states...)))
# V_predict = [V_func([x0,y0]) for y0 in ys for x0 in xs]
# dVdt_predict  = [V̇_func([x0,y0]) for y0 in ys for x0 in xs]

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

p1 = plot(
    xs, 
    ys, 
    V_predict, 
    linetype = 
    :contourf, 
    title = "V", 
    xlabel = "θ", 
    ylabel = "ω",
    c = :heat
    );
p1 = scatter!([-2*pi, 0, 2*pi], [0, 0, 0], label = "Stable Equilibria", color=:green, markershape=:+);
p1 = scatter!([-pi, pi], [0, 0], label = "Unstable Equilibria", color=:red, markershape=:x);
p2 = plot(
    xs,
    ys,
    dVdt_predict,
    linetype = :contourf,
    title = "dV/dt",
    xlabel = "θ",
    ylabel = "ω",
    c = :bluesreds
);
p2 = scatter!([-2*pi, 0, 2*pi], [0, 0, 0], label = "Stable Equilibria", color=:green, markershape=:+);
p2 = scatter!([-pi, pi], [0, 0], label = "Unstable Equilibria", color=:red, markershape=:x, legend=false);
p3 = plot(
    xs,
    ys,
    V_predict .< ρ,
    linetype = :contourf,
    title = "Estimated RoA",
    xlabel = "θ",
    ylabel = "ω",
    colorbar = false,
);
p4 = plot(
    xs,
    ys,
    dVdt_predict .< 0,
    linetype = :contourf,
    title = "dV/dt < 0",
    xlabel = "θ",
    ylabel = "ω",
    colorbar = false,
);
p4 = scatter!([-2*pi, 0, 2*pi], [0, 0, 0], label = "Stable Equilibria", color=:green, markershape=:+);
p4 = scatter!([-pi, pi], [0, 0], label = "Unstable Equilibria", color=:red, markershape=:x, legend=false);
plot(p1, p2, p4)


########################### Generate training data ############################
xs, ys = [(ub[i] - lb[i]) * rand(20) .+ lb[i] for i in eachindex(lb)]
data = [([x, y], f_true([x, y], p, 0.0)) for y in ys for x in xs]

######################### Define approximate dynamics #########################
Random.seed!(200)
function small_angle_approx(x, p, t)
    θ, ω = x
    ζ, ω_0 = p
    [ω;
    -2ζ * ω - ω_0^2 * θ]
end
approx_dynamics = ODEFunction(
        small_angle_approx; 
        syms = [:θ, :ω], 
        paramsyms = [:ζ, :ω_0]
    )

############################# Construct PDESystem #############################

pde_system, network_func = NeuralLyapunovPDESystem(
    approx_dynamics,
    lb,
    ub,
    spec;
    p = p
)

####################### Specify neural Lyapunov problem #######################
# Specify neural net
dim_state = length(lb)
dim_hidden = 15
dim_output = 2
chain = [
    Lux.Chain(
        Dense(dim_state, dim_hidden, tanh),
        Dense(dim_hidden, dim_hidden, tanh),
        Dense(dim_hidden, 1, use_bias = false),
    ) for _ = 1:dim_output
]

# Generate addtional loss
data_loss = additional_loss_from_data(data, spec, network_func; fixed_point = [0.0])

# Define neural network discretization
strategy = GridTraining(0.1)
discretization = PhysicsInformedNN(chain, strategy, additional_loss = data_loss)

######################## Construct OptimizationProblem ########################

prob = discretize(pde_system, discretization)
sym_prob = symbolic_discretize(pde_system, discretization)

########################## Solve OptimizationProblem ##########################

res = Optimization.solve(prob, Adam(); callback = callback, maxiters = 300)

prob = Optimization.remake(prob, u0 = res.u);
res = Optimization.solve(prob, Adam(); callback = callback, maxiters = 300)

println("Switching from Adam to BFGS");
prob = Optimization.remake(prob, u0 = res.u);
res = Optimization.solve(prob, BFGS(); callback = callback, maxiters = 300)

###################### Get numerical numerical functions ######################
V_func, V̇_func, ∇V_func = NumericalNeuralLyapunovFunctions(
    discretization.phi, 
    res, 
    network_func, 
    structure.V,
    true_dynamics,
    zeros(length(lb));
    p = p
    )

################################## Simulate ###################################
xs = lb[1]:0.02:ub[1] 
V_predict = [V_func([x0]) for x0 in xs]
dVdt_predict  = [V̇_func([x0]) for x0 in xs]

# Print statistics
println("V(0.,0.) = ", V_func([0.0]))
println("dVdt(0.,0.) = ", V̇_func([0.0]))
println("V ∋ [", min(V_func([0.0]), minimum(V_predict)), ", ", maximum(V_predict), "]")
println(
    "V̇ ∋ [",
    minimum(dVdt_predict),
    ", ",
    max(V̇_func([0.0]), maximum(dVdt_predict)),
    "]",
)

# Get RoA Estimate
invalid_region = xs[dVdt_predict .>= 0]
invalid_start = maximum(invalid_region[invalid_region .< 0])
invalid_end = minimum(invalid_region[invalid_region .> 0])
valid_region = xs[invalid_start .< xs .< invalid_end]
ρ = min(V_func([first(valid_region)]), V_func([last(valid_region)]))
RoA = valid_region[vec(V_func(transpose(valid_region))) .≤ ρ]

println("Estimated region of attraction: [$(first(RoA)), $(last(RoA))]")
println("True region of attraction: (-1, 1)")

# Plot results
p1 = plot(xs, V_predict, label = "V", xlabel = "x", linewidth=2);
p1 = hline!([ρ], label = "V = $(round(ρ, digits = 4))", legend = :inside)
p1 = vspan!([first(RoA), last(RoA)]; label = "Estimated Region of Attraction", opacity = 0.2, color = :green);
p1 = vspan!([-1, 1]; label = "True Region of Attraction", opacity = 0.2, color = :gray);

p2 = plot(xs, dVdt_predict, label = "dV/dt", xlabel = "x", linewidth=2);
p2 = hline!([0.0], label = "dV/dt = 0", legend = :bottom)
p2 = vspan!([first(RoA), last(RoA)]; label = "Estimated Region of Attraction", opacity = 0.2, color = :green);
p2 = vspan!([-1, 1]; label = "True Region of Attraction", opacity = 0.2, color = :gray);

plot(p1, p2)