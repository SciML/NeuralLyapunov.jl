module NeuralLyapunov

import ForwardDiff
using ModelingToolkit
using LinearAlgebra
using Optimization
import OptimizationOptimJL
import Hypatia, JuMP

include("conditions_specification.jl")
include("NeuralLyapunovPDESystem.jl")
include("local_Lyapunov.jl")
include("incorporate_data.jl")

export NeuralLyapunovPDESystem, NumericalNeuralLyapunovFunctions
export local_Lyapunov
export NeuralLyapunovSpecification, NeuralLyapunovStructure, 
    UnstructuredNeuralLyapunov, NonnegativeNeuralLyapunov, 
    LyapunovMinimizationCondition, StrictlyPositiveDefinite, 
    PositiveSemiDefinite, DontCheckNonnegativity, LyapunovDecreaseCondition,
    AsymptoticDecrease, ExponentialDecrease, DontCheckDecrease,
    additional_loss_from_data

end
