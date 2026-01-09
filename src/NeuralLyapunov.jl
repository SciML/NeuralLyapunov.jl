module NeuralLyapunov

import ForwardDiff
import JuMP
using LinearAlgebra: I, dot, ⋅
import Symbolics
using Symbolics: @variables, Equation, Num, diff2term, value
using ModelingToolkit: @named, @parameters, ODESystem, PDESystem, parameters, unknowns,
    defaults, operation, unbound_inputs, defaults, structural_simplify
import SciMLBase
using SciMLBase: ODEFunction, ODEInputFunction, ODEProblem, solve, EnsembleProblem,
    EnsembleThreads, remake

using SymbolicIndexingInterface: SymbolCache, variable_symbols
using NeuralPDE: PhysicsInformedNN, discretize, LogOptions
import NeuralPDE
using OrdinaryDiffEq: AutoTsit5, Rosenbrock23
import LuxCore
using Lux: Chain, Parallel, NoOpLayer, WrappedFunction, f16, f32, f64
using MLDataDevices: cpu_device
using Boltz.Layers: ShiftTo
using StableRNGs: StableRNG
using QuasiMonteCarlo: sample, LatinHypercubeSample
using DataFrames: DataFrame

const cpud = cpu_device()

include("conditions_specification.jl")
include("structure_specification.jl")
include("minimization_conditions.jl")
include("decrease_conditions.jl")
include("decrease_conditions_RoA_aware.jl")
include("NeuralLyapunovPDESystem.jl")
include("numerical_lyapunov_functions.jl")
include("local_lyapunov.jl")
include("policy_search.jl")
include("logger.jl")
include("benchmark_harness.jl")
include("lux_structures.jl")

# Lyapunov function structures
export NeuralLyapunovStructure, NoAdditionalStructure, NonnegativeStructure,
    PositiveSemiDefiniteStructure, get_numerical_lyapunov_function

# Lux structures
export AdditiveLyapunovNet, MultiplicativeLyapunovNet, SoSPooling,
    StrictlyPositiveSoSPooling

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
