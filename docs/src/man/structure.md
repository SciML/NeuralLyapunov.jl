# Structuring a neural Lyapunov function

```@docs
NeuralLyapunovStructure
```

## Pre-defined structures

Three simple neural Lyapunov function structures are provided, but users can always specify a custom structure using the [`NeuralLyapunovStructure`](@ref) struct.

The simplest structure is to train the neural network directly to be the Lyapunov function, which can be accomplished using an [`UnstructuredNeuralLyapunov`](@ref).

```@docs
UnstructuredNeuralLyapunov
```

The condition that the Lyapunov function ``V(x)`` must be minimized uniquely at the fixed point ``x_0`` is often represented as a requirement for ``V(x)`` to be positive away from the fixed point and zero at the fixed point.
Put mathematically, it is sufficient to require ``V(x) > 0 \, \forall x \ne x_0`` and ``V(x_0) = 0``.
We call such functions positive definite (around the fixed point ``x_0``).

Two structures are provided which partially or fully enforce the minimization condition: [`NonnegativeNeuralLyapunov`](@ref), which structurally enforces ``V(x) \ge 0`` everywhere, and [`PositiveSemiDefiniteStructure`](@ref), which additionally enforces ``V(x_0) = 0``.

```@docs
NonnegativeNeuralLyapunov
PositiveSemiDefiniteStructure
```
