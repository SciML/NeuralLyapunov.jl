using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, Plots

# Define parameters and differentials
@parameters x y
@variables u1(..) u2(..)
Dx = Differential(x)
Dy = Differential(y)
divergence(F) = Dx(F[1]) + Dy(F[2])
curl(F) = Dy(F[1]) - Dx(F[2])

# Define PDE system
us = [u1(x,y), u2(x,y)]
eqs = [ divergence(us) ~ 0,
        curl(us) ~ 0
        ]
domains = [ x ∈ (0.0, 1.0),
            y ∈ (0.0, 1.0) 
            ]
bcs = [ u1(x,0) ~ cos(x), 
        u2(x,0) ~ sin(x),
        u1(x,1) ~ cos(x)*exp(1),
        u2(x,1) ~ sin(x)*exp(1),
        u1(0,y) ~ exp(y),
        u2(0,y) ~ 0.,
        u1(1,y) ~ cos(1)*exp(y),
        u2(1,y) ~ sin(1)*exp(y)
        ]

@named pde_system = PDESystem(eqs, bcs, domains, [x, y], us)

# Define neural network discretization
dim_input = length(domains)
dim_hidden = 15
chain = [Lux.Chain(
                Dense(dim_input, dim_hidden, Lux.σ), 
                Dense(dim_hidden, dim_hidden, Lux.σ),
                Dense(dim_hidden, 1)
                )
            for _ in 1:length(us)
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
minimizers_ = [res.u.depvar[Symbol(:u,i)] for i in 1:length(us)]
predicted_sol_func(x,y) = [ phi[i]([x,y],minimizers_[i])[1] for i in 1:length(us) ]

# Plot results
xs,ys = [ModelingToolkit.infimum(d.domain):0.01:ModelingToolkit.supremum(d.domain) for d in domains]

analytic_sol_func(x,y) = [exp(y)*cos(x), exp(y)*sin(x)]
u_real = [[analytic_sol_func(x,y)[i] for x in xs for y in ys] for i in 1:length(us)]
u_predict = [[predicted_sol_func(x,y)[i] for x in xs for y in ys] for i in 1:length(us)]
diff_u = [abs.(u_real[i] .- u_predict[i]) for i in 1:length(us)]

for i in 1:length(us)
    p1 = plot(xs, ys, u_real[i], linetype=:contourf,title = "u$i, analytic");
    p2 = plot(xs, ys, u_predict[i], linetype=:contourf,title = "u$i, predict");
    p3 = plot(xs, ys, diff_u[i], linetype=:contourf,title = "u$i, error");
    plot(p1,p2,p3)
    savefig("harmonic_sol_u$i")
end