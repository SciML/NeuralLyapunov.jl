using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimisers, OptimizationOptimJL, Plots, ForwardDiff, LinearAlgebra, NLopt

# Define parameters and differentials
@parameters x y
@variables u1(..) u2(..)
grad(f) = Symbolics.gradient(f, [x, y])

# Define Lyapunov function
dim_output = 2
u(x0,y0) = Num.([u1(x0,y0), u2(x0,y0)])
δ = 0.01
V_sym(x0,y0) = (u(x0,y0) - u(0.,0.)) ⋅ (u(x0,y0) - u(0.,0.)) + δ*log(1. + x0^2 + y0^2)
#V_sym(x0,y0) = (u(x0,y0)) ⋅ (u(x0,y0)) + δ*log(1. + x0^2 + y0^2)

# Define dynamics and Lyapunov conditions
dynamics(x0,y0) = [y0; -y0-x0]
V̇_sym(x0, y0) = dynamics(x0,y0) ⋅ grad(V_sym(x0,y0))
eq = max(0., V̇_sym(x, y)) ~ 0.
κ = 20.
#eq = log(1. + exp( κ * V̇_sym(x,y))) ~ 0. # Stricter, but max(0, V̇) still trains fine
domains = [ x ∈ (-2*pi, 2*pi),
            y ∈ (-10., 10.) 
            ]
bcs = [ V_sym(0.,0.) ~ 0. ] 

# Construct PDESystem
@named pde_system = PDESystem(eq, bcs, domains, [x, y], u(x,y))

# Define neural network discretization
dim_input = length(domains)
dim_hidden = 15
chain = [Lux.Chain(
                Dense(dim_input, dim_hidden, tanh), 
                Dense(dim_hidden, dim_hidden, tanh),
                Dense(dim_hidden, 1, use_bias=false)
                )
            for _ in 1:dim_output
            ]

#strategy = QuadratureTraining()
strategy = GridTraining(0.05)
#strategy = QuasiRandomTraining(1000, bcs_points=1)
#strategy = StochasticTraining(1000, bcs_points=1)
discretization = PhysicsInformedNN(chain, strategy)
prob = discretize(pde_system, discretization)

callback = function (p, l)
    println("loss: ", l)
    return false
end

# Solve 
#opt = BFGS()
opt = Adam()
opt = AdaGrad()
opt = AdaMax()
opt = Optim.SimulatedAnnealing()
res = Optimization.solve(prob, opt; callback=callback, maxiters=1000)
phi = discretization.phi

u_func(x0,y0) = [ phi[i]([x0,y0], res.u.depvar[Symbol(:u,i)])[1] for i in 1:dim_output ]

function V_func(x0,y0) 
    u_vec = u_func(x0,y0) - u_func(0.,0.)
#    u_vec = u_func(x0,y0)
    norm(u_vec)^2 + δ*log(1 + x0^2 + y0^2)
end

∇V_func(x0,y0) = ForwardDiff.gradient(p -> V_func(p[1], p[2]), [x0, y0])
V̇_func(x0,y0) = dynamics(x0,y0) ⋅ ∇V_func(x0,y0)

# Plot results
xs,ys = [ModelingToolkit.infimum(d.domain):0.01:ModelingToolkit.supremum(d.domain) for d in domains]
V_predict = [V_func(x0,y0) for y0 in ys for x0 in xs]
dVdt_predict  = [V̇_func(x0,y0) for y0 in ys for x0 in xs]
p1 = plot(xs, ys, V_predict, linetype=:contourf, title = "V", xlabel="x", ylabel="ẋ");
p2 = plot(xs, ys, dVdt_predict, linetype=:contourf, title="dV/dt", xlabel="x", ylabel="ẋ");
#p2 = scatter!([-pi, pi], [0., 0.], label="Unstable equilibria");
#p2 = scatter!([-2*pi, 0., 2*pi], [0., 0., 0.], label="Stable equilibria");
plot(p1, p2)
savefig("Lyapunov_sol")

println("V(0.,0.) = ", V_func(0.,0.))
println("V ∈ [", min(V_func(0.,0.), minimum(V_predict)), ", ", maximum(V_predict), "]")
println("V̇ ∈ [", minimum(dVdt_predict), ", ", max(V̇_func(0.,0.), maximum(dVdt_predict)), "]")
