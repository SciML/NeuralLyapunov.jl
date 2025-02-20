using ModelingToolkit
import ModelingToolkit: inputs, generate_control_function
using NeuralLyapunovProblemLibrary
using OrdinaryDiffEq
using Test
using Plots

############### Undriven double pendulum should drop to downward equilibrium ###############
x0 = vcat(2π * rand(2) .- π, rand(2))

# Assume uniform rods of random mass and length
m1, m2 = rand(2)
l1, l2 = rand(2)
lc1, lc2 = l1 /2, l2 / 2
I1 = m1 * l1^2 / 3
I2 = m2 * l2^2 / 3
g = 1.0
p = [I1, I2, l1, l2, lc1, lc2, m1, m2, g]

prob = ODEProblem(structural_simplify(double_pendulum_undriven), x0, 100, p)
sol = solve(prob, Tsit5())

anim = plot_pendulum(sol)
@test anim isa Plots.Animation
# gif(anim, fps=50)
