module NeuralLyapunov

import ForwardDiff
import JuMP
using LinearAlgebra: I, dot, ⋅
import Symbolics
using Symbolics: @variables, Equation, Num, diff2term, value
using ModelingToolkit: @named, @parameters, ODESystem, PDESystem, parameters, unknowns,
                       defaults, operation, unbound_inputs, generate_control_function,
                       get_defaults
import SciMLBase
using SciMLBase: ODEFunction, ODEProblem, EnsembleProblem, EnsembleThreads, solve
using SymbolicIndexingInterface: SymbolCache, variable_symbols
using NeuralPDE: PhysicsInformedNN, discretize
using OrdinaryDiffEq: Tsit5
using EvalMetrics: ConfusionMatrix
import LuxCore
using StableRNGs: StableRNG
using QuasiMonteCarlo: sample, LatinHypercubeSample
using Optimization: remake

include("conditions_specification.jl")
include("structure_specification.jl")
include("minimization_conditions.jl")
include("decrease_conditions.jl")
include("decrease_conditions_RoA_aware.jl")
include("NeuralLyapunovPDESystem.jl")
include("numerical_lyapunov_functions.jl")
include("local_lyapunov.jl")
include("policy_search.jl")
include("benchmark_harness.jl")

# Lyapunov function structures
export NeuralLyapunovStructure, UnstructuredNeuralLyapunov, NonnegativeNeuralLyapunov,
       PositiveSemiDefiniteStructure, get_numerical_lyapunov_function

# Minimization conditions
export LyapunovMinimizationCondition, StrictlyPositiveDefinite, PositiveSemiDefinite,
       DontCheckNonnegativity

# Decrease conditions
export LyapunovDecreaseCondition, StabilityISL, AsymptoticStability, ExponentialStability,
       DontCheckDecrease

# Setting up the PDESystem for NeuralPDE
export NeuralLyapunovSpecification, NeuralLyapunovPDESystem

# Region of attraction handling
export RoAAwareDecreaseCondition, make_RoA_aware

# Policy search
export add_policy_search, get_policy

# Local Lyapunov analysis
export local_lyapunov

# Benchmarking tool
export benchmark

end
