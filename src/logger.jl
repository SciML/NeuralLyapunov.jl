"""

    NeuralLyapunovBenchmarkLogger(losses, iterations)

A simple logger for tracking the (full weighted) loss during training using NeuralPDE.

# Fields
  - `losses::Vector{<:Real}`: A vector to store the loss values.
  - `iterations::Vector{<:Integer}`: A vector to store the corresponding iteration numbers.

# Constructors
  - `NeuralLyapunovBenchmarkLogger{T1, T2}() where {T1<:Real, T2<:Integer}`: Creates an
    empty logger with specified types for losses and iterations.
  - `NeuralLyapunovBenchmarkLogger{T}() where {T}`: Creates an empty logger with specified
    type for losses and `Int64` for iterations.
  - `NeuralLyapunovBenchmarkLogger()`: Creates an empty logger with `Float64` for losses and
    `Int64` for iterations.
"""
struct NeuralLyapunovBenchmarkLogger{T1 <: Real, T2 <: Integer}
    losses::Vector{T1}
    iterations::Vector{T2}
end

function NeuralLyapunovBenchmarkLogger{T1, T2}() where {T1, T2}
    return NeuralLyapunovBenchmarkLogger{T1, T2}(T1[], T2[])
end
NeuralLyapunovBenchmarkLogger{T}() where {T} = NeuralLyapunovBenchmarkLogger{T, Int64}()
NeuralLyapunovBenchmarkLogger() = NeuralLyapunovBenchmarkLogger{Float64}()

function NeuralPDE.logscalar(logger::NeuralLyapunovBenchmarkLogger, scalar::Real,
        name::AbstractString, step::Integer)
    if name == "weighted_loss/full_weighted_loss"
        push!(logger.losses, scalar)
        push!(logger.iterations, step)
    end
    return nothing
end
