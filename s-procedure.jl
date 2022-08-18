using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, Plots, ForwardDiff, LinearAlgebra

# Define parameters and differentials
@parameters x y
@variables u1(..) u2(..) l1(..) l2(..)
grad(f) = Symbolics.gradient(f, [x, y])

# Define Lyapunov function
u_dim = 2
l_dim = 2
u(x0,y0) = Num.([u1(x0,y0), u2(x0,y0)])
l(x0,y0) = Num.([l1(x0,y0), l2(x0,y0)])
δ = 0.01
#V_sym(x0,y0) = (u(x0,y0) - u(0.,0.)) ⋅ (u(x0,y0) - u(0.,0.)) + δ*log(1. + x0^2 + y0^2)
V_sym(x0,y0) = (u(x0,y0)) ⋅ (u(x0,y0)) + δ*log(1. + x0^2 + y0^2)
λ_sym(x0,y0) = (l(x0,y0)) ⋅ (l(x0,y0)) + δ*log(1. + x0^2 + y0^2)

# Define dynamics and Lyapunov conditions
dynamics(x0,y0) = [y0; -y0-sin(x0)]
V̇_sym(x0, y0) = dynamics(x0,y0) ⋅ grad(V_sym(x0,y0))

#eq = max(0., dynamics(x,y) ⋅ grad(V_sym(x,y))) ~ 0.
eqs = [ log(1. + exp(10.0*(V̇_sym(x, y) + λ_sym(x,y) * (V_sym(x,y) - 1) ))) ~ 0.,
        λ_sym(x,y) ~ 0. # I think this helps maximize the area of the RoA?
        ]
domains = [ x ∈ (-2*pi, 2*pi),
            y ∈ (-10., 10.) 
            ]
bcs = [ V_sym(0.,0.) ~ 0.,
        λ_sym(0.,0.) ~ 0.
        ] 

# Construct PDESystem
@named pde_system = PDESystem(eqs, bcs, domains, [x, y], vcat(u(x,y), l(x,y)))

# Define neural network discretization
dim_input = length(domains)
dim_hidden = 8
chain = [Lux.Chain(
                Dense(dim_input, dim_hidden, Lux.σ), 
                Dense(dim_hidden, dim_hidden, Lux.σ),
                Dense(dim_hidden, 1)
                )
            for _ in 1:(u_dim + l_dim)
            ]

#strategy = QuadratureTraining()
#strategy = GridTraining(0.05)
strategy = QuasiRandomTraining(1000, bcs_points=1)
#strategy = StochasticTraining(1000, bcs_points=1)
discretization = PhysicsInformedNN(chain, strategy)
prob = discretize(pde_system, discretization)

callback = function (p, l)
    println("loss: ", l)
    return false
end

# Solve 
res = Optimization.solve(prob, BFGS(); callback=callback, maxiters=100)
phi = discretization.phi

u_func(x0,y0) = [ phi[i]([x0,y0], res.u.depvar[Symbol(:u,i)])[1] for i in 1:u_dim ]

V_func(x0, y0) = norm(u_func(x0,y0))^2 + δ*log(1+x0^2+y0^2)
#V_func(x0, y0) = norm(u_func(x0,y0) - u_func(0.,0.))^2 + δ*log(1+x0^2+y0^2)
∇V_func(x0,y0) = ForwardDiff.gradient(p -> V_func(p[1], p[2]), [x0, y0])
V̇_func(x0,y0) = dynamics(x0,y0) ⋅ ∇V_func(x0,y0)

l_func(x0,y0) = [ phi[i]([x0,y0], res.u.depvar[Symbol(:l,i)])[1] for i in 1:l_dim ]
λ_func(x0, y0) = norm(l_func(x0,y0))^2 + δ*log(1+x0^2+y0^2)

# Plot results
xs,ys = [ModelingToolkit.infimum(d.domain):0.01:ModelingToolkit.supremum(d.domain) for d in domains]
V_predict = [V_func(x0,y0) for y0 in ys for x0 in xs]
dVdt_predict  = [V̇_func(x0,y0) for y0 in ys for x0 in xs]
λ_predict = [λ_func(x0,y0) for y0 in ys for x0 in xs]
p1 = plot(xs, ys, V_predict, linetype=:contourf, title = "V", xlabel="x", ylabel="ẋ");
p2 = plot(xs, ys, dVdt_predict, linetype=:contourf, title="dV/dt", xlabel="x", ylabel="ẋ");
p3 = plot(xs, ys, λ_predict, linetype=:contourf, title = "λ", xlabel="x", ylabel="ẋ");
plot(p1, p2, p3)
savefig("S-Lyapunov_sol")

println("V(0.,0.) = ", V_func(0.,0.))
println("V ∈ [", min(V_func(0.,0.), minimum(V_predict)), ", ", maximum(V_predict), "]")
println("V̇ ∈ [", minimum(dVdt_predict), ", ", maximum(dVdt_predict), "]")
println("λ(0.,0.) = ", λ_func(0.,0.))
println("λ ∈ [", min(λ_func(0.,0.), minimum(λ_predict)), ", ", maximum(λ_predict), "]")
