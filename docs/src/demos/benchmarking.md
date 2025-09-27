# Benchmarking a neural Lyapunov method

In this demonstration, we'll benchmark the neural Lyapunov method used in the [policy search demo](policy_search.md).
In that demonstration, we searched for a neural network policy to stabilize the upright equilibrium of the inverted pendulum.
Here, we will use the [`benchmark`](@ref) function to run approximately the same training, then check the performance the of the resulting controller and neural Lyapunov function by simulating the closed loop system to see (1) how well the controller drives the pendulum to the upright equilibrium, and (2) how well the neural Lyapunov function performs as a classifier of whether a state is in the region of attraction or not.
These results will be represented by a confusion matrix using the simulation results as ground truth.
(Keep in mind that training does no simulation.)

## Copy-Pastable Code

```julia
using NeuralPDE, NeuralLyapunov, Lux
using Boltz.Layers: PeriodicEmbedding
import OptimizationOptimisers, OptimizationOptimJL
using StableRNGs, Random

rng = StableRNG(0)
Random.seed!(200)

# Define dynamics and domain
function open_loop_pendulum_dynamics(x, u, p, t)
    θ, ω = x
    ζ, ω_0 = p
    τ = u[]
    return [ω, -2ζ * ω_0 * ω - ω_0^2 * sin(θ) + τ]
end

lb = [0.0, -2.0];
ub = [2π, 2.0];
upright_equilibrium = [π, 0.0]
p = Float32[0.5, 1.0]
state_syms = [:θ, :ω]
parameter_syms = [:ζ, :ω_0]

# Define neural network discretization
# We use an input layer that is periodic with period 2π with respect to θ
dim_state = length(lb)
dim_hidden = 25
dim_phi = 3
dim_u = 1
dim_output = dim_phi + dim_u
chain = [Chain(
             PeriodicEmbedding([1], Float32[2π]),
             Dense(dim_state + 1, dim_hidden, tanh),
             Dense(dim_hidden, dim_hidden, tanh),
             Dense(dim_hidden, 1)
         ) for _ in 1:dim_output]
ps, st = Lux.setup(rng, chain)

# Define neural network discretization
strategy = QuasiRandomTraining(10000)

# Define neural Lyapunov structure
periodic_pos_def = function (state, fixed_point)
    θ, ω = state
    θ_eq, ω_eq = fixed_point
    return (sin(θ) - sin(θ_eq))^2 + (cos(θ) - cos(θ_eq))^2 + 0.1 * (ω - ω_eq)^2
end

structure = PositiveSemiDefiniteStructure(
    dim_phi;
    pos_def = (x, x0) -> log(1.0 + periodic_pos_def(x, x0))
)
structure = add_policy_search(structure, dim_u)

minimization_condition = DontCheckNonnegativity(check_fixed_point = false)

# Define Lyapunov decrease condition
decrease_condition = AsymptoticStability(strength = periodic_pos_def)

# Construct neural Lyapunov specification
spec = NeuralLyapunovSpecification(structure, minimization_condition, decrease_condition)

# Define optimization parameters
opt = [OptimizationOptimisers.Adam(0.05), OptimizationOptimJL.BFGS()]
optimization_args = [[:maxiters => 300], [:maxiters => 300]]

# Run benchmark
endpoint_check = (x) -> ≈([sin(x[1]), cos(x[1]), x[2]], [0, -1, 0], atol = 5e-3)
benchmarking_results = benchmark(
    open_loop_pendulum_dynamics,
    lb,
    ub,
    spec,
    chain,
    strategy,
    opt;
    simulation_time = 200,
    n = 1000,
    fixed_point = upright_equilibrium,
    p,
    optimization_args,
    state_syms,
    parameter_syms,
    policy_search = true,
    endpoint_check,
    init_params = ps, 
    init_states = st
)
```

## Detailed Description

Much of the set up is the same as in the [policy search demo](policy_search.md), so see that page for details.


```@example benchmarking
using NeuralPDE, NeuralLyapunov, Lux
import Boltz.Layers: PeriodicEmbedding
using Random, StableRNGs

Random.seed!(200)

# Define dynamics and domain
function open_loop_pendulum_dynamics(x, u, p, t)
    θ, ω = x
    ζ, ω_0 = p
    τ = u[]
    return [ω, -2ζ * ω_0 * ω - ω_0^2 * sin(θ) + τ]
end

lb = [0.0, -2.0];
ub = [2π, 2.0];
upright_equilibrium = [π, 0.0]
p = Float32[0.5, 1.0]
state_syms = [:θ, :ω]
parameter_syms = [:ζ, :ω_0]

# Define neural network discretization
# We use an input layer that is periodic with period 2π with respect to θ
dim_state = length(lb)
dim_hidden = 25
dim_phi = 3
dim_u = 1
dim_output = dim_phi + dim_u
chain = [Chain(
             PeriodicEmbedding([1], Float32[2π]),
             Dense(3, dim_hidden, tanh),
             Dense(dim_hidden, dim_hidden, tanh),
             Dense(dim_hidden, 1)
         ) for _ in 1:dim_output]
ps, st = Lux.setup(StableRNG(0), chain)

# Define neural network discretization
strategy = QuasiRandomTraining(10000)

# Define neural Lyapunov structure
periodic_pos_def = function (state, fixed_point)
    θ, ω = state
    θ_eq, ω_eq = fixed_point
    return (sin(θ) - sin(θ_eq))^2 + (cos(θ) - cos(θ_eq))^2 + 0.1 * (ω - ω_eq)^2
end

structure = PositiveSemiDefiniteStructure(
    dim_phi;
    pos_def = (x, x0) -> log(1.0 + periodic_pos_def(x, x0))
)
structure = add_policy_search(structure, dim_u)

minimization_condition = DontCheckNonnegativity(check_fixed_point = false)

# Define Lyapunov decrease condition
decrease_condition = AsymptoticStability(strength = periodic_pos_def)

# Construct neural Lyapunov specification
spec = NeuralLyapunovSpecification(structure, minimization_condition, decrease_condition)
nothing # hide
```

At this point of the [policy search demo](policy_search.md), we constructed the PDESystem, discretized it using NeuralPDE.jl, and solved the resulting OptimizationProblem using Optimization.jl.
All of that occurs in the [`benchmark`](@ref) function, so we instead provide that function with the optimizer and optimization arguments to use.

```@example benchmarking
import OptimizationOptimisers, OptimizationOptimJL

# Define optimization parameters
opt = [OptimizationOptimisers.Adam(0.05), OptimizationOptimJL.BFGS()]
optimization_args = [[:maxiters => 300], [:maxiters => 300]]
nothing # hide
```

Since the pendulum is periodic in ``0``, we'll use a custom endpoint check that reflects that property.

```@example benchmarking
endpoint_check = (x) -> ≈([sin(x[1]), cos(x[1]), x[2]], [0, -1, 0], atol=5e-3)
nothing # hide
```

Finally, we can run the [`benchmark`](@ref) function.
For demonstration purposes, we'll use `EnsembleSerial()`, which simulates each trajectory without any parallelism when evaluating the trained Lyapunov function and controller.
The default `ensemble_alg` is `EnsembleThreads()`, which uses multithreading (local parallelism only); see the [DifferentialEquations.jl docs](https://docs.sciml.ai/DiffEqDocs/stable/features/ensemble/) for more information and other options.
Another option is [`EnsembleGPUArray`](https://docs.sciml.ai/DiffEqGPU/stable/manual/ensemblegpuarray/), which parallelizes the ODE solves on the GPU.
Note that this option is imported from `DiffEqGPU` and has certain restrictions on the dynamics.
For example, the dynamics may not allocate memory (build arrays), so in-place dynamics must be defined in addition to the out-of-place dynamics that NeuralLyapunov usually requires.
(Providing both can be achieved by defining both methods for the same function and passing in an `ODEFunction` made from that function.)
For this reason, the default value of `EnsembleThreads()` is recommended, even when training occurs on GPU.

```@example benchmarking
using OrdinaryDiffEq: EnsembleSerial

benchmarking_results = benchmark(
    open_loop_pendulum_dynamics,
    lb,
    ub,
    spec,
    chain,
    strategy,
    opt;
    simulation_time = 200,
    n = 1000,
    fixed_point = upright_equilibrium,
    p,
    optimization_args,
    state_syms,
    parameter_syms,
    policy_search = true,
    ensemble_alg = EnsembleSerial(),
    endpoint_check,
    init_params = ps, 
    init_states = st
);
nothing # hide
```

We can observe the confusion matrix and training time:

```@example benchmarking
benchmarking_results.confusion_matrix
```

```@example benchmarking
benchmarking_results.training_time
```

The `benchmark` function also outputs a `DataFrame`, `data`, with the simulation results.
The first three rows are shown below.

```@example benchmarking
benchmarking_results.data[1:3, :]
```

The `benchmark` function also outputs the Lyapunov function ``V`` and its time-derivative ``V̇``.

```@example benchmarking
states = benchmarking_results.data[!, "Initial State"]
V_samples = benchmarking_results.data[!, "V"]
all(benchmarking_results.V.(states) .== V_samples)
```

```@example benchmarking
V̇_samples = benchmarking_results.data[!, "dVdt"]
all(benchmarking_results.V̇.(states) .== V̇_samples)
```

The "Actually in RoA" column is just the result of applying `endpoint_check` applied to the "End State" column.
The "End State" column is the final state of the simulation starting at that "Initial State".

```@example benchmarking
endpoints = benchmarking_results.data[!, "Final State"]
actual = benchmarking_results.data[!, "Actually in RoA"]
all(endpoint_check.(endpoints) .== actual)
```

Similarly, the labels in the "Predicted in RoA" column are the results of the neural Lyapunov classifier.

```@example benchmarking
classifier = (V, V̇, x) -> V̇ < zero(V̇) || endpoint_check(x)
predicted = benchmarking_results.data[!, "Predicted in RoA"]
all(classifier.(V_samples, V̇_samples, states) .== predicted)
```
