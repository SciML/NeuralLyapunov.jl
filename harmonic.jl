using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, Plots, ForwardDiff, LinearAlgebra

# Define parameters and differentials
@parameters x y
@variables u1(..) u2(..)
Dx = Differential(x)
Dy = Differential(y)
#divergence(F) = Dx(F[1]) + Dy(F[2])
divergence(F) = sum(diag(Symbolics.jacobian(F, [x, y])))
curl(F) = Dy(F[1]) - Dx(F[2])
#grad(f) = [Dx(f), Dy(f)]
grad(f) = Symbolics.gradient(f, [x, y])

# Define PDE system
us = [u1(x,y), u2(x,y)]
u(x0,y0) = sin(u1(x0,y0))*exp(u2(x0,y0))
eq = divergence(grad(u(x,y))) ~ 0
domains = [ x ∈ (0.0, pi),
            y ∈ (0.0, 1.0) 
            ]
bcs = [ u(x,0) ~ sin(x),
        u(x,1) ~ sin(x)*exp(1),
        u(0,y) ~ 0.,
        u(pi,y) ~ 0.
        ]

@named pde_system = PDESystem(eq, bcs, domains, [x, y], us)

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

#TODO: Only QuasiRandomTraining works
#strategy = QuadratureTraining()
#strategy = GridTraining(0.05)
strategy = QuasiRandomTraining(100)
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
us_predict_func(x0,y0) = [ phi[i]([x0,y0],minimizers_[i])[1] for i in 1:length(us) ]
#=function predicted_sol_func(x0,y0) 
    u_vec = u_predict_func(x0,y0)
    sin(u_vec[1]) * exp(u_vec[2])
end=#
predicted_sol_func = ( (s) -> sin(s[1])*exp(s[2]) ) ∘ us_predict_func

# Plot results
xs,ys = [ModelingToolkit.infimum(d.domain):0.01:ModelingToolkit.supremum(d.domain) for d in domains]

analytic_sol_func(x0,y0) = exp(y0)*sin(x0)
u_real = [analytic_sol_func(x0,y0) for y0 in ys for x0 in xs]
u_predict = [predicted_sol_func(x0,y0) for y0 in ys for x0 in xs]
diff_u = abs.(u_real .- u_predict)

p1 = plot(xs, ys, u_real, linetype=:contourf,title = "u, analytic");
p2 = plot(xs, ys, u_predict, linetype=:contourf,title = "u, predict");
p3 = plot(xs, ys, diff_u, linetype=:contourf,title = "u, error");

plot(p1,p2,p3)
savefig("harmonic_sol_u")

function grad_fun(x0,y0)
    ForwardDiff.gradient(p -> predicted_sol_func(p[1], p[2]), [x0, y0])
end

function lap_fun(x0,y0)
    sum(diag(ForwardDiff.hessian(p -> predicted_sol_func(p[1], p[2]), [x0, y0])))
end

function curl_fun(x0,y0)
    H = ForwardDiff.hessian(p -> predicted_sol_func(p[1], p[2]), [x0, y0])
    H[1,2] - H[2,1]
end

grad_predict = [norm(grad_fun(x0,y0)) for y0 in ys for x0 in xs]
lap_predict  = [lap_fun(x0,y0) for y0 in ys for x0 in xs]
curl_predict = [curl_fun(x0,y0) for y0 in ys for x0 in xs]


p1 = plot(xs, ys, lap_predict, linetype=:contourf, title="laplacian");
p2 = plot(xs, ys, curl_predict, linetype=:contourf, title="curl of grad");
p3 = plot(xs, ys, grad_predict, linetype=:contourf, title="grad magnitude");
plot(p3, p1, p2)
savefig("harmonic_err")

#V_fun(x,v) = norm(φ_fun(x,v) - φ_fun(0.,0.))^2 + δ * log(1.0 + x^2 + v^2) 
#dVdt_fun(x0,v0) = Symbolics.value.(substitute(f, Dict([x=>x0, v=>v0]))) ⋅ gradient(V_fun, x0, v0)