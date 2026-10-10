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

Optimization.jl callback that records the training loss in `logger` whenever the training
iteration is a multiple of `log_frequency`.

The training iteration is the solver's own iteration counter (`state.iter`) plus the number
of iterations completed in earlier optimization passes; call `start_pass!` before
each `solve` so that training with several optimizers in sequence produces one continuous
loss history. Repeated callbacks for an iteration that has already been seen (e.g., the
final callback that OptimizationOptimisers issues with `save_best = true`, or the initial
iteration `0` that Optim.jl reports at the start of a pass) are ignored.
"""
mutable struct LossLoggingCallback{L <: NeuralLyapunovBenchmarkLogger}
    logger::L
    log_frequency::Int
    pass_offset::Int
    pass_iterations::Int
    last_iteration::Int
end

function LossLoggingCallback(logger, log_frequency)
    return LossLoggingCallback(logger, log_frequency, 0, 0, typemin(Int))
end

"""
    start_pass!(cb::LossLoggingCallback)

Mark the start of a new optimization pass: iterations reported by the next `solve` are
counted after the last iteration of the previous pass.
"""
function start_pass!(cb::LossLoggingCallback)
    cb.pass_offset += cb.pass_iterations
    cb.pass_iterations = 0
    return cb
end

function (cb::LossLoggingCallback)(state, loss)
    iteration = cb.pass_offset + state.iter
    iteration > cb.last_iteration || return false
    cb.last_iteration = iteration
    cb.pass_iterations = max(cb.pass_iterations, state.iter)
    if iteration % cb.log_frequency == 0
        log_loss!(cb.logger, loss, iteration)
    end
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
