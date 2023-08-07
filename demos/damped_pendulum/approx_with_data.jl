using LinearAlgebra
using NeuralPDE, Lux, ModelingToolkit
using Optimization, OptimizationOptimisers, OptimizationOptimJL, NLopt
using Plots
using NeuralLyapunov
using Random

###############################################################################
############### First, run with full knowledge of true dynamics ###############
###############################################################################
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
p = [0.5, 5.0]
true_dynamics = ODEFunction(f_true; syms = [:θ, :ω], paramsyms = [:ζ, :ω_0])

####################### Specify neural Lyapunov problem #######################

# Define neural network discretization
# We use an input layer that is periodic with period 2π with respect to θ
dim_state = length(lb)
dim_hidden = 15
dim_output = 2
chain_true = [
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
discretization_true = PhysicsInformedNN(chain_true, strategy)

# Define neural Lyapunov structure
structure = PositiveSemiDefiniteStructure(
        dim_output;
        pos_def = function (state, fixed_point)
            θ, ω = state
            θ_eq, ω_eq = fixed_point
            log(1.0 + (sin(θ)-sin(θ_eq))^2 + (cos(θ)-cos(θ_eq))^2 + (ω-ω_eq)^2)
        end
    )
minimization_condition = DontCheckNonnegativity(check_fixed_point = false)

# Define Lyapunov decrease condition
decrease_condition = AsymptoticDecrease(strict = true)

# Construct neural Lyapunov specification
spec = NeuralLyapunovSpecification(
    structure,
    minimization_condition,
    decrease_condition,
    )

############################# Construct PDESystem #############################

pde_system_true, network_func = NeuralLyapunovPDESystem(
    true_dynamics,
    lb,
    ub,
    spec;
    p = p
)

######################## Construct OptimizationProblem ########################

prob_true = discretize(pde_system_true, discretization_true)
sym_prob_true = symbolic_discretize(pde_system_true, discretization_true)

callback = function (p, l)
    println("loss: ", l)
    return false
end

########################## Solve OptimizationProblem ##########################

res_true = Optimization.solve(prob_true, Adam(); callback = callback, maxiters = 300)

prob_true = Optimization.remake(prob_true, u0 = res_true.u);
res_true = Optimization.solve(prob_true, Adam(); callback = callback, maxiters = 300)

println("Switching from Adam to BFGS");
prob_true = Optimization.remake(prob_true, u0 = res_true.u);
res_true = Optimization.solve(prob_true, BFGS(); callback = callback, maxiters = 300)

###################### Get numerical numerical functions ######################
V_func_true, V̇_func_true, _ = NumericalNeuralLyapunovFunctions(
    discretization_true.phi, 
    res_true, 
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
V_predict_true = vec(V_func_true(hcat(states...)))
dVdt_predict_true = vec(V̇_func_true(hcat(states...)))

# Print statistics
println("V(0.,0.) = ", V_func_true([0.0, 0.0]))
println("V ∋ [", min(V_func_true([0.0, 0.0]), minimum(V_predict_true)), ", ", maximum(V_predict_true), "]")
println(
    "V̇ ∋ [",
    minimum(dVdt_predict_true),
    ", ",
    max(V̇_func_true([0.0, 0.0]), maximum(dVdt_predict_true)),
    "]",
)

# Plot results

p1 = plot(
    xs/pi, 
    ys, 
    V_predict_true, 
    linetype = 
    :contourf, 
    title = "V", 
    xlabel = "θ/π", 
    ylabel = "ω",
    c = :bone_1
    );
p1 = scatter!([-2*pi, 0, 2*pi]/pi, [0, 0, 0], label = "Stable Equilibria", color=:green, markershape=:+);
p1 = scatter!([-pi, pi]/pi, [0, 0], label = "Unstable Equilibria", color=:red, markershape=:x);
p2 = plot(
    xs/pi,
    ys,
    dVdt_predict_true,
    linetype = :contourf,
    title = "dV/dt",
    xlabel = "θ/π",
    ylabel = "ω",
    c = :binary
);
p2 = scatter!([-2*pi, 0, 2*pi]/pi, [0, 0, 0], label = "Stable Equilibria", color=:green, markershape=:+);
p2 = scatter!([-pi, pi]/pi, [0, 0], label = "Unstable Equilibria", color=:red, markershape=:x, legend=false);
p3 = plot(
    xs/pi,
    ys,
    dVdt_predict_true .< 0,
    linetype = :contourf,
    title = "dV/dt < 0",
    xlabel = "θ/π",
    ylabel = "ω",
    colorbar = false,
    linewidth = 0
);
p3 = scatter!([-2*pi, 0, 2*pi]/pi, [0, 0, 0], label = "Stable Equilibria", color=:green, markershape=:+);
p3 = scatter!([-pi, pi]/pi, [0, 0], label = "Unstable Equilibria", color=:red, markershape=:x, legend=false);
plot(p1, p2, p3)


###############################################################################
################## Then, run only with approximate dynamics ###################
###############################################################################
Random.seed!(200)

######################### Define approximate dynamics #########################
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

pde_system_approx, network_func = NeuralLyapunovPDESystem(
    approx_dynamics,
    lb,
    ub,
    spec;
    p = p
)

####################### Specify neural Lyapunov problem #######################

# Define neural network discretization
# We use an input layer that is periodic with period 2π with respect to θ
chain_approx = [
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
discretization_approx = PhysicsInformedNN(chain_approx, strategy)

######################## Construct OptimizationProblem ########################

prob_approx = discretize(pde_system_approx, discretization_approx)
sym_prob_approx = symbolic_discretize(pde_system_approx, discretization_approx)

########################## Solve OptimizationProblem ##########################

res_approx = Optimization.solve(prob_approx, Adam(); callback = callback, maxiters = 300)

prob_approx = Optimization.remake(prob_approx, u0 = res_approx.u);
res_approx = Optimization.solve(prob_approx, Adam(); callback = callback, maxiters = 300)

println("Switching from Adam to BFGS");
prob_approx = Optimization.remake(prob_approx, u0 = res_approx.u);
res_approx = Optimization.solve(prob_approx, BFGS(); callback = callback, maxiters = 300)

###################### Get numerical numerical functions ######################
V_func_approx, V̇_func_approx, _ = NumericalNeuralLyapunovFunctions(
    discretization_approx.phi, 
    res_approx, 
    network_func, 
    structure.V,
    true_dynamics,
    zeros(length(lb));
    p = p
    )

################################## Simulate ###################################
V_predict_approx = vec(V_func_approx(hcat(states...)))
dVdt_predict_approx = vec(V̇_func_approx(hcat(states...)))

# Print statistics
println("V(0.,0.) = ", V_func_approx([0.0, 0.0]))
println("V ∋ [", min(V_func_approx([0.0, 0.0]), minimum(V_predict_approx)), ", ", maximum(V_predict_approx), "]")
println(
    "V̇ ∋ [",
    minimum(dVdt_predict_approx),
    ", ",
    max(V̇_func_approx([0.0, 0.0]), maximum(dVdt_predict_approx)),
    "]",
)

# Plot results
p4 = plot(
    xs/pi, 
    ys, 
    V_predict_approx, 
    linetype = 
    :contourf, 
    title = "V", 
    xlabel = "θ/π", 
    ylabel = "ω",
    c = :bone_1
    );
p4 = scatter!([-2*pi, 0, 2*pi]/pi, [0, 0, 0], label = "Stable Equilibria", color=:green, markershape=:+);
p4 = scatter!([-pi, pi]/pi, [0, 0], label = "Unstable Equilibria", color=:red, markershape=:x);
p5 = plot(
    xs/pi,
    ys,
    dVdt_predict_approx,
    linetype = :contourf,
    title = "dV/dt",
    xlabel = "θ/π",
    ylabel = "ω",
    c = :binary
);
p5 = scatter!([-2*pi, 0, 2*pi]/pi, [0, 0, 0], label = "Stable Equilibria", color=:green, markershape=:+);
p5 = scatter!([-pi, pi]/pi, [0, 0], label = "Unstable Equilibria", color=:red, markershape=:x, legend=false);
p6 = plot(
    xs/pi,
    ys,
    dVdt_predict_approx .< 0,
    linetype = :contourf,
    title = "dV/dt < 0",
    xlabel = "θ/π",
    ylabel = "ω",
    colorbar = false,
    linewidth = 0
);
p6 = scatter!([-2*pi, 0, 2*pi]/pi, [0, 0, 0], label = "Stable Equilibria", color=:green, markershape=:+);
p6 = scatter!([-pi, pi]/pi, [0, 0], label = "Unstable Equilibria", color=:red, markershape=:x, legend=false);
plot(p4, p5, p6)

###############################################################################
############ Finally, run with approximate dynamics and true data #############
###############################################################################


########################### Generate training data ############################
data_xs, data_ys = [(ub[i] - lb[i]) * rand(20) .+ lb[i] for i in eachindex(lb)]
data = [([x, y], f_true([x, y], p, 0.0)) for y in data_ys for x in data_xs]
Random.seed!(200)

####################### Specify neural Lyapunov problem #######################

# Define neural network discretization
# We use an input layer that is periodic with period 2π with respect to θ
chain_data = [
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

# Generate addtional loss
data_loss = additional_loss_from_data(
        data, 
        spec, 
        network_func; 
        fixed_point = [0.0, 0.0]
    )

# Define neural network discretization
discretization_data = PhysicsInformedNN(
        chain_data, 
        strategy, 
        additional_loss=data_loss
    )

######################## Construct OptimizationProblem ########################

prob_data = discretize(pde_system_approx, discretization_data)
sym_prob_data = symbolic_discretize(pde_system_approx, discretization_data)

########################## Solve OptimizationProblem ##########################

res_data = Optimization.solve(prob_data, Adam(); callback = callback, maxiters = 300)

prob_data = Optimization.remake(prob_data, u0 = res_data.u);
res_data = Optimization.solve(prob_data, Adam(); callback = callback, maxiters = 300)

println("Switching from Adam to BFGS");
prob_data = Optimization.remake(prob_data, u0 = res_data.u);
res_data = Optimization.solve(prob_data, BFGS(); callback = callback, maxiters = 300)

###################### Get numerical numerical functions ######################
V_func_data, V̇_func_data, _ = NumericalNeuralLyapunovFunctions(
    discretization_data.phi, 
    res_data, 
    network_func, 
    structure.V,
    true_dynamics,
    zeros(length(lb));
    p = p
    )

################################## Simulate ###################################
V_predict_data = vec(V_func_data(hcat(states...)))
dVdt_predict_data = vec(V̇_func_data(hcat(states...)))

# Print statistics
println("V(0.,0.) = ", V_func_data([0.0, 0.0]))
println("V ∋ [", min(V_func_data([0.0, 0.0]), minimum(V_predict_data)), ", ", maximum(V_predict_data), "]")
println(
    "V̇ ∋ [",
    minimum(dVdt_predict_data),
    ", ",
    max(V̇_func_data([0.0, 0.0]), maximum(dVdt_predict_data)),
    "]",
)

# Plot results
p7 = plot(
    xs/pi, 
    ys, 
    V_predict_data, 
    linetype = 
    :contourf, 
    title = "V", 
    xlabel = "θ/π", 
    ylabel = "ω",
    c = :bone_1
    );
p7 = scatter!([-2*pi, 0, 2*pi]/pi, [0, 0, 0], label = "Stable Equilibria", color=:green, markershape=:+);
p7 = scatter!([-pi, pi]/pi, [0, 0], label = "Unstable Equilibria", color=:red, markershape=:x);
p8 = plot(
    xs/pi,
    ys,
    dVdt_predict_data,
    linetype = :contourf,
    title = "dV/dt",
    xlabel = "θ/π",
    ylabel = "ω",
    c = :binary
);
p8 = scatter!([-2*pi, 0, 2*pi]/pi, [0, 0, 0], label = "Stable Equilibria", color=:green, markershape=:+);
p8 = scatter!([-pi, pi]/pi, [0, 0], label = "Unstable Equilibria", color=:red, markershape=:x, legend=false);
p9 = plot(
    xs/pi,
    ys,
    dVdt_predict_data .< 0,
    linetype = :contourf,
    title = "dV/dt < 0",
    xlabel = "θ/π",
    ylabel = "ω",
    colorbar = false,
    linewidth = 0
);
p9 = scatter!([-2*pi, 0, 2*pi]/pi, [0, 0, 0], label = "Stable Equilibria", color=:green, markershape=:+);
p9 = scatter!([-pi, pi]/pi, [0, 0], label = "Unstable Equilibria", color=:red, markershape=:x, legend=false);
plot(p7, p8, p9)
