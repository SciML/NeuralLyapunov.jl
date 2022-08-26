using NeuralPDE, Lux, Optimization, OptimizationOptimisers, Plots

@parameters x
@variables u(..)

# Set up equation
v(x0) = u(x0) - u(0.)
#v(x0) = u(x0)
eq  = v(x) ~ sin(x)

# Set up domain
x_min, x_max = -pi, pi
domains = [x ∈ (x_min,x_max)]

# Set up boundary condition
bcs = [ v(0.) ~ 0. ]

# Neural network
hidden_dim = 16
chain = Lux.Chain(
                Dense(1,hidden_dim,Lux.σ),
                Dense(hidden_dim,hidden_dim,Lux.σ),
                Dense(hidden_dim,1)
                )

# Discretization
dx = 0.05
strategy = GridTraining(dx)
discretization = PhysicsInformedNN(chain, strategy)

@named pde_system = PDESystem(eq,bcs,domains,[x],[u(x)])
prob = discretize(pde_system,discretization)
symprob = symbolic_discretize(pde_system, discretization)

#Optimizer
opt = Adam()

#Callback function
callback = function (p,l)
    println("Current loss is: $l")
    return false
end

res = Optimization.solve(prob, opt, callback = callback, maxiters=20)
phi = discretization.phi

xs = x_min:dx/10:x_max
analytic_sol_func(x) = sin(x)
predcit_sol_func(x) = first(phi(x,res.u)) - first(phi(0.,res.u))
#predcit_sol_func(x) = first(phi(x,res.u))
u_predict = [ predcit_sol_func(x) for x in xs]
u_real = [analytic_sol_func(x) for x in xs]
diff_u = abs.(u_predict .- u_real)

plot(xs, u_real,label = "analytic", color = :blue)
plot!(xs, u_predict,label = "predict", color = :gray)
plot!(xs, diff_u,label = "error", color = :red)
