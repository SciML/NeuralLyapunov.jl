# Lyapunov Decrease Condition

To represent the condition that the Lyapunov function ``V(x)`` must decrease along system trajectories, we typically introduce a new function ``\dot{V}(x) = \nabla_{f(x)} V(x) = \nabla_x V(x) \cdot f(x)``.
It is then sufficient to show that ``\dot{V}(x)`` is negative away from the fixed point and zero at the fixed point, since ``\dot{V}`` represents the rate of change of ``V`` along system trajectories.
i.e., if ``x(t)`` is a trajectory defined by ``\frac{dx}{dt} = f(x)``, then ``\dot{V}(x(t)) = \frac{d}{dt} [ V(x(t)) ]``.

Put mathematically, it is sufficient to require ``\dot{V}(x) < 0 \, \forall x \ne x_0`` and ``\dot{V}(x_0) = 0``.
We call such functions negative definite (around the fixed point ``x_0``).
The weaker condition that ``\dot{V}(x) \le 0 \, \forall x \ne x_0`` and ``\dot{V}(x_0) = 0`` is negative *semi-*definiteness.

The condition that ``\dot{V}(x_0) = 0`` is satisfied by the definition of ``\dot{V}`` and the fact that ``x_0`` is a fixed point, so we do not need to train for it unless the dynamics have some dependence on the neural network (e.g., in [policy search](policy_search.md)).

NeuralLyapunov.jl provides the [`LyapunovDecreaseCondition`](@ref) struct for users to specify the form of the decrease condition to enforce through training.

```@docs
LyapunovDecreaseCondition
```

## Pre-defined decrease conditions

```@docs
AsymptoticDecrease
ExponentialDecrease
DontCheckDecrease
```

## Defining your own decrease condition

```@docs
NeuralLyapunov.AbstractLyapunovDecreaseCondition
```

```@docs
NeuralLyapunov.check_decrease
NeuralLyapunov.get_decrease_condition
```
