using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, Plots, ForwardDiff, LinearAlgebra

# Define parameters and differentials
@parameters x y
@variables u1(..) u2(..)
grad(f) = Symbolics.gradient(f, [x, y])

# Define Lyapunov function
dim_output = 2
u(x0,y0) = Num.([u1(x0,y0), u2(x0,y0)])
δ = 0.01
V_sym(x0,y0) = u(x0,y0) ⋅ u(x0,y0) + δ*log(1. + x0^2 + y0^2)

# Define dynamics and Lyapunov conditions
dynamics(x0,y0) = [y0; -y0-x0]
eq = max(0., dynamics(x,y) ⋅ grad(V_sym(x,y))) ~ 0.
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
                Dense(dim_input, dim_hidden, Lux.σ), 
                Dense(dim_hidden, dim_hidden, Lux.σ),
                Dense(dim_hidden, 1)
                )
            for _ in 1:dim_output
            ]

#TODO: Only QuasiRandomTraining works
#strategy = QuadratureTraining()
#strategy = GridTraining(0.1)
strategy = QuasiRandomTraining(100, bcs_points=10)
discretization = PhysicsInformedNN(chain, strategy)
prob = discretize(pde_system, discretization)

callback = function (p, l)
    println("loss: ", l)
    return false
end

# Solve 
res = Optimization.solve(prob, BFGS(); callback=callback, maxiters=5000)
phi = discretization.phi
minimizers_ = [res.u.depvar[Symbol(:u,i)] for i in 1:dim_output]

u_predict_func(x0,y0) = [ phi[i]([x0,y0],minimizers_[i])[1] for i in 1:dim_output ]

function V_func(x0,y0) 
    u_vec = u_predict_func(x0,y0)# - us_predict_func(0.,0.)
    norm(u_vec)^2 + δ*log(1 + x0^2 + y0^2)
end

∇V_fun(x0,y0) = ForwardDiff.gradient(p -> V_func(p[1], p[2]), [x0, y0])
V̇_func(x0,y0) = dynamics(x0,y0) ⋅ ∇V_fun(x0,y0)

# Plot results
xs,ys = [ModelingToolkit.infimum(d.domain):0.01:ModelingToolkit.supremum(d.domain) for d in domains]
V_predict = [V_func(x0,y0) for y0 in ys for x0 in xs]
dVdt_predict  = [V̇_func(x0,y0) for y0 in ys for x0 in xs]
p1 = plot(xs, ys, V_predict, linetype=:contourf, title = "V", xlabel="x", ylabel="ẋ");
p2 = plot(xs, ys, dVdt_predict, linetype=:contourf, title="dV/dt", xlabel="x", ylabel="ẋ");
plot(p1, p2)#, p3, p4)
savefig("Lyapunov_sol")

println("V(0.,0.) = ", V_func(0.,0.))
println("V ∈ [", minimum(V_predict), ", ", maximum(V_predict), "]")
println("V̇ ∈ [", minimum(dVdt_predict), ", ", maximum(dVdt_predict), "]")

#grad_predict = [norm(grad_fun(x0,y0)) for y0 in ys for x0 in xs]
#lap_predict  = [lap_fun(x0,y0) for y0 in ys for x0 in xs]

#curl_predict = [curl_fun(x0,y0) for y0 in ys for x0 in xs]


#p3 = plot(xs, ys, grad_predict, linetype=:contourf, title="grad magnitude");
#p4 = plot(xs, ys, curl_predict, linetype=:contourf, title="curl of grad");
