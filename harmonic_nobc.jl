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
δ = 0.01
V(x0,y0) = (u1(x0,y0))^2 + (u2(x0,y0))^2 + δ*log(1. + x0^2 + y0^2)
eq = divergence(grad(V(x,y))) ~ 0
domains = [ x ∈ (0.0, pi),
            y ∈ (0.0, 1.0) 
            ]
bcs = [ V(0.5, 0.5) ~ 1. ]

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
strategy = QuasiRandomTraining(500)
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
function predicted_sol_func(x0,y0) 
    u_vec = us_predict_func(x0,y0)
    u_vec[1]^2 + u_vec[2]^2 + δ*log(1 + x0^2 + y0^2)
end
#predicted_sol_func = ( (s) -> s[1]^2 + s[2]^2 ) ∘ us_predict_func

# Plot results
xs,ys = [ModelingToolkit.infimum(d.domain):0.01:ModelingToolkit.supremum(d.domain) for d in domains]
u_predict = [predicted_sol_func(x0,y0) for y0 in ys for x0 in xs]

p1 = plot(xs, ys, u_predict, linetype=:contourf,title = "u, predict");

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

#grad_predict = [norm(grad_fun(x0,y0)) for y0 in ys for x0 in xs]
lap_predict  = [lap_fun(x0,y0) for y0 in ys for x0 in xs]
#curl_predict = [curl_fun(x0,y0) for y0 in ys for x0 in xs]


p2 = plot(xs, ys, lap_predict, linetype=:contourf, title="laplacian");
#p3 = plot(xs, ys, grad_predict, linetype=:contourf, title="grad magnitude");
#p4 = plot(xs, ys, curl_predict, linetype=:contourf, title="curl of grad");
plot(p1, p2)#, p3, p4)
savefig("harmonic_sol_nobc")

println("V(0.5,0.5) = ", predicted_sol_func(0.5,0.5))
println("V ∈ (", minimum(u_predict), ", ", maximum(u_predict), ")")
println("ΔV ∈ (", minimum(lap_predict), ", ", maximum(lap_predict), ")")

#V_fun(x,v) = norm(φ_fun(x,v) - φ_fun(0.,0.))^2 + δ * log(1.0 + x^2 + v^2) 
#dVdt_fun(x0,v0) = Symbolics.value.(substitute(f, Dict([x=>x0, v=>v0]))) ⋅ gradient(V_fun, x0, v0)