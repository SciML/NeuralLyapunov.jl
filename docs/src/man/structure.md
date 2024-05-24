# Structuring a neural Lyapunov function

Three simple neural Lyapunov function structures are provided, but users can always specify a custom structure using the [`NeuralLyapunovStructure`](@ref) struct.

## Pre-defined structures

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

## Defining your own neural Lyapunov function structure

To define a new structure for a neural Lyapunov function, one must specify the form of the Lyapunov candidate ``V`` and its time derivative along a trajectory ``\dot{V}``, as well as how to call the dynamics ``f``.
Additionally, the dimensionality of the output of the neural network must be known in advance.

```@docs
NeuralLyapunovStructure
```

### Calling the dynamics

Very generally, the dynamical system can be a system of ODEs ``\dot{x} = f(x, u, p, t)``, where ``u`` is a control input, ``p`` contains parameters, and ``f`` depends on the neural network in some way.
To capture this variety, users must supply the function `f_call(dynamics::Function, phi::Function, state, p, t)`.

The most common example is when `dynamics` takes the same form as an `ODEFunction`. 
i.e., ``\dot{x} = \texttt{dynamics}(x, p, t)``.
In that case, `f_call(dynamics, phi, state, p, t) = dynamics(state, p, t)`.

Suppose instead, the dynamics were supplied as a function of state alone: ``\dot{x} = \texttt{dynamics}(x)``.
Then, `f_call(dynamics, phi, state, p, t) = dynamics(state)`.

Finally, suppose ``\dot{x} = \texttt{dynamics}(x, u, p, t)`` has a unidimensional control input that is being trained (via [policy search](policy_search.md)) to be the second output of the neural network.
Then `f_call(dynamics, phi, state, p, t) = dynamics(state, phi(state)[2], p, t)`.

Note that, despite the inclusion of the time variable ``t``, NeuralLyapunov.jl currently only supports time-invariant systems, so only `t = 0.0` is used.

### Structuring ``V`` and ``\dot{V}``

The Lyapunov candidate function ``V`` gets specified as a function `V(phi, state, fixed_point)`, where `phi` is the neural network as a function `phi(state)`.
Note that this form allows ``V(x)`` to depend on the neural network evaluated at points other than just the input ``x``.

The time derivative ``\dot{V}`` is similarly defined by a function `V̇(phi, J_phi, dynamics, state, params, t, fixed_point)`.
The function `J_phi(state)` gives the Jacobian of the neural network `phi` at `state`.
The function `dynamics` is as above (with parameters `params`). 
