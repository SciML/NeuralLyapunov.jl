module NeuralLyapunov

import ForwardDiff
import JuMP
using LinearAlgebra
using ModelingToolkit
import SciMLBase

include("conditions_specification.jl")
include("structure_specification.jl")
include("minimization_conditions.jl")
include("decrease_conditions.jl")
include("decrease_conditions_RoA_aware.jl")
include("NeuralLyapunovPDESystem.jl")
include("numerical_lyapunov_functions.jl")
include("local_lyapunov.jl")
include("policy_search.jl")

# Lyapunov function structures
export NeuralLyapunovStructure, UnstructuredNeuralLyapunov, NonnegativeNeuralLyapunov,
       PositiveSemiDefiniteStructure, get_numerical_lyapunov_function

# Minimization conditions
export LyapunovMinimizationCondition, StrictlyPositiveDefinite, PositiveSemiDefinite,
       DontCheckNonnegativity

# Decrease conditions
export LyapunovDecreaseCondition, AsymptoticDecrease, ExponentialDecrease, DontCheckDecrease

# Setting up the PDESystem for NeuralPDE
export NeuralLyapunovSpecification, NeuralLyapunovPDESystem

# Region of attraction handling
export RoAAwareDecreaseCondition, make_RoA_aware

# Policy search
export add_policy_search, get_policy

# Local Lyapunov analysis
export local_lyapunov

end
