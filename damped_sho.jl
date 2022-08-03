using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, Plots

# Define parameters and differentials
@parameters x y
@variables u1(..) u2(..) u3(..)
Dx = Differential(x)
Dxx = Dx^2
Dy = Differential(y)
Dyy = Dy^2

# Define PDE system
eqs = [ Dyy(u1(x,y)) ~ Dxx(u1(x,y)) + u3(x,y)*sin(pi*x),
        Dyy(u2(x,y)) ~ Dxx(u2(x,y)) + u3(x,y)*cos(pi*x),
        0. ~ u1(x,y)*sin(pi*x) + u2(x,y)*cos(pi*x) - exp(-y) 
        ]
domains = [ x ∈ (0.0, 1.0),
            y ∈ (0.0, 1.0) 
            ]
bcs = [ u1(x,0) ~ sin(pi*x), 
        u2(x,0) ~ cos(pi*x),
        Dy(u1(x,0)) ~ -sin(pi*x), 
        Dy(u2(x,1)) ~ -cos(pi*x),
        u1(0,y) ~ 0.,
        u2(0,y) ~ exp(-y),
        u1(1,y) ~ 0.,
        u2(1,y) ~ -exp(-y) 
        ]

@named pde_system = PDESystem(eqs, bcs, domains, [x, y], [u1(x,y), u2(x,y), u3(x,y)])

# Define neural network discretization
dim_input = length(domains)
dim_hidden = 15
chain = [Lux.Chain(
                Dense(dim_input, dim_hidden, Lux.σ), 
                Dense(dim_hidden, dim_hidden, Lux.σ),
                Dense(dim_hidden, 1)
                )
            for _ in 1:3 
            ]

strategy = QuadratureTraining()
discretization = PhysicsInformedNN(chain, strategy)
prob = discretize(pde_system, discretization)

callback = function (p, l)
    println("loss: ", l)
    return false
end

# Solve 
res = Optimization.solve(prob, BFGS(); callback=callback, maxiters=5000)
phi = discretization.phi
minimizers_ = [res.u.depvar[Symbol(:u,i)] for i in 1:3]
predicted_sol_func(x,y) = [ phi[i]([x,y],minimizers_[i])[1] for i in 1:3 ]

# Plot results
xs,ys = [ModelingToolkit.infimum(d.domain):0.01:ModelingToolkit.supremum(d.domain) for d in domains]

analytic_sol_func(x,y) = [exp(-y)*sin(pi*x), exp(-y)*cos(pi*x), (1+pi^2)*exp(-y)]
u_real = [[analytic_sol_func(x,y)[i] for x in xs for y in ys] for i in 1:3]
u_predict = [[predicted_sol_func(x,y)[i] for x in xs for y in ys] for i in 1:3]
diff_u = [abs.(u_real[i] .- u_predict[i]) for i in 1:3]

for i in 1:3
    p1 = plot(xs, ys, u_real[i], linetype=:contourf,title = "u$i, analytic");
    p2 = plot(xs, ys, u_predict[i], linetype=:contourf,title = "u$i, predict");
    p3 = plot(xs, ys, diff_u[i], linetype=:contourf,title = "u$i, error");
    plot(p1,p2,p3)
    savefig("sol_u$i")
end