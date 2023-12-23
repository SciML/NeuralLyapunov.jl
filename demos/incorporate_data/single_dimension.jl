using LinearAlgebra
using NeuralPDE, Lux, ModelingToolkit
using Optimization, OptimizationOptimisers, OptimizationOptimJL, NLopt
using Plots
using NeuralLyapunov
using Random

Random.seed!(200)

######################### Define dynamics and domain ##########################

f_true(x, p, t) = -x .+ x.^3
lb = [-2.0];
ub = [2.0];
true_dynamics = ODEFunction(f_true; syms = [:x])

####################### Specify neural Lyapunov problem #######################

# Define neural network discretization
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

# Define neural network discretization
strategy = GridTraining(0.1)

# Define neural Lyapunov structure
structure = PositiveSemiDefiniteStructure(dim_output)
minimization_condition = DontCheckNonnegativity(check_fixed_point = true)

# Define Lyapunov decrease condition
decrease_condition = AsymptoticDecrease(strict = true)

# Construct neural Lyapunov specification
spec = NeuralLyapunovSpecification(
        structure,
        minimization_condition,
        decrease_condition,
    )

callback = function (p, l)
    println("loss: ", l)
    return false
end

########################### Generate training data ############################
data_xs = (ub[1] - lb[1]) * rand(20) .+ lb[1]
data = [([x], f_true(x, [], 0.0)) for x in data_xs]

simulation_xs = lb[1]:0.02:ub[1] 

######################### Define approximate dynamics #########################
f_approx(x, p, t) = -x
approx_dynamics = ODEFunction(f_approx; syms = [:x])

########################### Loop through experiments ##########################

struct LyapunovResults
    θ
    V::Function
    V̇::Function
    ρ::Real
    RoA::Tuple{Real, Real}
    plt
end

results = Dict{Symbol, LyapunovResults}()

for experiment in [:true_dynamics, :approx_no_data, :approx_with_data]
    Random.seed!(200)
    println("\nBeginning experiment ", experiment)

    ######################## Construct PDESystem ##########################

    pde_system, network_func = if experiment == :true_dynamics
        NeuralLyapunovPDESystem(
            true_dynamics,
            lb,
            ub,
            spec
        )
    else
        NeuralLyapunovPDESystem(
            approx_dynamics,
            lb,
            ub,
            spec
        )
    end

    ####################### Generate discretization #######################

    discretization = if experiment == :approx_with_data
        # Generate addtional loss
        data_loss = additional_loss_from_data(
            data, 
            spec, 
            network_func; 
            fixed_point = [0.0]
        )
        PhysicsInformedNN(chain, strategy, additional_loss = data_loss)
    else
        PhysicsInformedNN(chain, strategy)
    end

    #################### Construct OptimizationProblem ####################

    prob = discretize(pde_system, discretization)
    sym_prob = symbolic_discretize(pde_system, discretization)

    ###################### Solve OptimizationProblem ######################

    res = Optimization.solve(prob, Adam(); callback = callback, maxiters = 300)

    prob = Optimization.remake(prob, u0 = res.u);
    res = Optimization.solve(prob, Adam(); callback = callback, maxiters = 300)

    println("Switching from Adam to BFGS");
    prob = Optimization.remake(prob, u0 = res.u);
    res = Optimization.solve(prob, BFGS(); callback = callback, maxiters = 300)

    ################## Get numerical numerical functions ##################
    V_func, V̇_func, _ = NumericalNeuralLyapunovFunctions(
            discretization.phi, 
            res.u, 
            network_func, 
            structure.V,
            true_dynamics,
            zeros(length(lb))
        )

    ############################# Save results ############################

    results[experiment] = LyapunovResults(
        res.u, 
        V_func, 
        V̇_func, 
        0.0, 
        (0.0, 0.0),
        nothing
        )
end

for experiment in [:true_dynamics, :approx_no_data, :approx_with_data]
    V_func = results[experiment].V
    V̇_func = results[experiment].V̇

    ############################## Simulate ###############################
    V_predict = [V_func([x]) for x in simulation_xs]
    dVdt_predict  = [V̇_func([x]) for x in simulation_xs]

    # Print statistics
    println("V(0.,0.) = ", V_func([0.0]))
    println("dVdt(0.,0.) = ", V̇_func([0.0]))
    println(
        "V ∋ [", 
        min(V_func([0.0]), 
        minimum(V_predict)), 
        ", ", 
        maximum(V_predict), "]"
    )
    println(
        "V̇ ∋ [",
        minimum(dVdt_predict),
        ", ",
        max(V̇_func([0.0]), maximum(dVdt_predict)),
        "]",
    )

    # Get RoA Estimate
    invalid_region = simulation_xs[dVdt_predict .>= 0]
    invalid_start = maximum(invalid_region[invalid_region .< 0])
    invalid_end = minimum(invalid_region[invalid_region .> 0])
    valid_region = simulation_xs[invalid_start .< simulation_xs .< invalid_end]
    ρ = min(V_func([first(valid_region)]), V_func([last(valid_region)]))
    RoA = valid_region[vec(V_func(transpose(valid_region))) .≤ ρ]

    # Plot results
    p_V = plot(simulation_xs, V_predict, label = "V", xlabel = "x", linewidth=2);
    p_V = hline!([ρ], label = "V = $(round(ρ, digits = 4))", legend = :top);
    p_V = vspan!([first(RoA), last(RoA)]; label = "Estimated Region of Attraction", opacity = 0.2, color = :green);
    p_V = vspan!([-1, 1]; label = "True Region of Attraction", color = :gray, fillstyle = :/);

    p_V̇ = plot(simulation_xs, dVdt_predict, label = "dV/dt", xlabel = "x", linewidth=2);
    p_V̇ = hline!([0.0], label = "dV/dt = 0", legend = :top);
    p_V̇ = vspan!([first(RoA), last(RoA)]; label = "Estimated Region of Attraction", opacity = 0.2, color = :green);
    p_V̇ = vspan!([-1, 1]; label = "True Region of Attraction", color = :gray, fillstyle = :/);

    plt_title = if experiment == :true_dynamics
        "Using True Dynamics"
    elseif experiment == :approx_no_data
        "Using Approximate Dynamics Only"
    else
        "Using Approximate Dynamics + Data"
    end
        
    plt = plot(p_V, p_V̇, plot_title=plt_title)

    results[experiment] = LyapunovResults(
        results[experiment].θ, 
        V_func, 
        V̇_func, 
        ρ, 
        (first(RoA), last(RoA)),
        plt
        )
end

println("\n\nSummary\n")
println("True region of attraction: (-1.0, 1.0)")
println("RoA estimate with true dynamics: ", results[:true_dynamics].RoA)
println("RoA estimate with approximate dynamics alone: ", results[:approx_no_data].RoA)
println("RoA estimate with approximate dynamics + data: ", results[:approx_with_data].RoA)

plot(results[:true_dynamics].plt)
plot(results[:approx_no_data].plt)
plot(results[:approx_with_data].plt)