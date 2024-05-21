# Lyapunov Minimization Condition

The condition that the Lyapunov function ``V(x)`` must be minimized uniquely at the fixed point ``x_0`` is often represented as a requirement for ``V(x)`` to be positive away from the fixed point and zero at the fixed point.
Put mathematically, it is sufficient to require ``V(x) > 0 \, \forall x \ne x_0`` and ``V(x_0) = 0``.
We call such functions positive definite (around the fixed point ``x_0``).
The weaker condition that ``V(x) \ge 0 \, \forall x \ne x_0`` and ``V(x_0) = 0`` is positive *semi-*definiteness.

Users specify the form of the minimization condition to enforce through training using a [`LyapunovMinimizationCondition`](@ref).

```@docs
LyapunovMinimizationCondition     
```

## Pre-defined minimization conditions

```@docs
PositiveSemiDefinite
DontCheckNonnegativity
StrictlyPositiveDefinite
```

## Defining your own minimization condition

```@docs
NeuralLyapunov.AbstractLyapunovMinimizationCondition
```

```@docs
NeuralLyapunov.check_nonnegativity
NeuralLyapunov.check_minimal_fixed_point
NeuralLyapunov.get_minimization_condition
```
