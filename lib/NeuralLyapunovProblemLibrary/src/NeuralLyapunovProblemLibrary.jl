module NeuralLyapunovProblemLibrary

using ModelingToolkit: @independent_variables, @parameters, System
using Symbolics: @variables, Differential
using SciMLBase: NullParameters
using LinearAlgebra: ×
using Rotations: RotZXY

include("pendulum.jl")
export Pendulum
export plot_pendulum

include("double_pendulum.jl")
export DoublePendulum, Acrobot, Pendubot
export plot_double_pendulum

include("quadrotor.jl")
export QuadrotorPlanar, Quadrotor3D
export plot_quadrotor_planar, plot_quadrotor_3d

end
