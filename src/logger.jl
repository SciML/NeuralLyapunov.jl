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

function log_loss!(logger::NeuralLyapunovBenchmarkLogger, loss::Real, iteration::Integer)
    push!(logger.losses, loss)
    push!(logger.iterations, iteration)
    return nothing
end

"""
    LossLoggingCallback(logger, log_frequency)

Optimization.jl callback that records the training loss in `logger` every `log_frequency`
iterations. Iterations are counted from `0` across every `solve` call that uses the callback,
so training with several optimizers in sequence produces one continuous loss history.
"""
mutable struct LossLoggingCallback{L <: NeuralLyapunovBenchmarkLogger}
    logger::L
    log_frequency::Int
    iteration::Int
end

LossLoggingCallback(logger, log_frequency) = LossLoggingCallback(logger, log_frequency, 0)

function (cb::LossLoggingCallback)(state, loss)
    if cb.iteration % cb.log_frequency == 0
        log_loss!(cb.logger, loss, cb.iteration)
    end
    cb.iteration += 1
    return false
end

"""
    with_training_callback(optimization_args, training_callback)

Return `optimization_args` with its `callback` replaced by one that first calls
`training_callback` and then any user-supplied `callback`. A vector of argument vectors (one
per optimization pass) is handled elementwise.
"""
function with_training_callback(optimization_args, training_callback)
    user_callback = nothing
    args = Pair{Symbol, Any}[]
    for (k, v) in optimization_args
        if Symbol(k) === :callback
            user_callback = v
        else
            push!(args, Symbol(k) => v)
        end
    end
    callback = function (state, loss, rest...)
        training_callback(state, loss)
        return user_callback === nothing ? false : user_callback(state, loss, rest...)
    end
    push!(args, :callback => callback)
    return args
end

function with_training_callback(
        optimization_args::AbstractVector{<:AbstractVector}, training_callback
    )
    return [with_training_callback(args, training_callback) for args in optimization_args]
end
