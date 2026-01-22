module NeuralLyapunovProblemLibrary

using ModelingToolkit: @parameters, ODESystem, compose, unbound_inputs, t_nounits as t,
    D_nounits as Dt
using Symbolics: @variables
using SciMLBase: NullParameters
using LinearAlgebra: ×
using Rotations: RotZXY

DDt = Dt^2

include("pendulum.jl")
export Pendulum, control_pendulum, get_pendulum_state_symbols, get_pendulum_param_symbols
export plot_pendulum

include("double_pendulum.jl")
export DoublePendulum, Acrobot, Pendubot, control_double_pendulum,
    get_double_pendulum_state_symbols, get_double_pendulum_param_symbols
export plot_double_pendulum

include("planar_quadrotor.jl")
export QuadrotorPlanar, control_quadrotor_planar, get_quadrotor_planar_state_symbols,
    get_quadrotor_planar_param_symbols, get_quadrotor_planar_input_symbols
export plot_quadrotor_planar

include("quadrotor_3d.jl")
export Quadrotor3D, control_quadrotor_3d, get_quadrotor_3d_state_symbols,
    get_quadrotor_3d_param_symbols, get_quadrotor_3d_input_symbols
export plot_quadrotor_3d

end
