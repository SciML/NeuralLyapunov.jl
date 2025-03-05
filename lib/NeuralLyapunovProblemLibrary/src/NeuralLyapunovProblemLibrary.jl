module NeuralLyapunovProblemLibrary

using ModelingToolkit
using LinearAlgebra
using Rotations: RotZXY

include("pendulum.jl")
export pendulum, pendulum_undriven
export plot_pendulum

include("double_pendulum.jl")
export double_pendulum, acrobot, pendubot, double_pendulum_undriven
export plot_double_pendulum

include("quadrotor.jl")
export quadrotor_planar, quadrotor_3d
export plot_quadrotor_planar, plot_quadrotor_3d

end
