# Structuring a Neural Lyapunov function

Users have two ways to specify the structure of their neural Lyapunov function: through Lux and through NeuralLyapunov.
NeuralLyapunov.jl is intended for use with [NeuralPDE.jl](https://github.com/SciML/NeuralPDE.jl), which is itself intended for use with [Lux.jl](https://github.com/LuxDL/Lux.jl).
Users therefore must provide a Lux model representing the neural network ``\phi(x)`` which will be trained, regardless of the transformation they use to go from ``\phi`` to ``V``.

In some cases, users will find it simplest to make ``\phi(x)`` a simple multilayer perceptron and specify their neural Lyapunov function structure as a transformation of the function ``\phi`` to the function ``V`` using the [`NeuralLyapunovStructure`](@ref) struct, detailed below.

In other cases (particularly when they find the NeuralPDE parser has trouble tracing their structure), users may wish to represent their neuralLyapunov function structure using [Lux.jl](https://github.com/LuxDL/Lux.jl)/[Boltz.jl](https://github.com/LuxDL/Boltz.jl) layers, integrating them into ``\phi(x)`` and letting ``V(x) = \phi(x)`` (as in [`NoAdditionalStructure`](@ref), detailed below).

Users may also combine the two methods, particularly if they find that their structure can be broken down into a component that the NeuralPDE parser has trouble tracing but exists in Lux/Boltz, and another aspect that can be written easily using a [`NeuralLyapunovStructure`](@ref) but does not correspond any existing Lux/Boltz layer.
(Such an example will be provided below.)

NeuralLyapunov.jl supplies two Lux structures and two pooling layers for structuring ``\phi(x)``, along with three [`NeuralLyapunovStructure`](@ref) transformations.
Additionally, users can always specify a custom structure using the [`NeuralLyapunovStructure`](@ref) struct.

## Pre-defined NeuralLyapunov transformations

The simplest structure is to train the neural network directly to be the Lyapunov function, which can be accomplished using an [`NoAdditionalStructure`](@ref).
This is particularly useful with the pre-defined Lux structures detailed in the following section.

```@docs
NoAdditionalStructure
```

The condition that the Lyapunov function ``V(x)`` must be minimized uniquely at the fixed point ``x_0`` is often represented as a requirement for ``V(x)`` to be positive away from the fixed point and zero at the fixed point.
Put mathematically, it is sufficient to require ``V(x) > 0 \, \forall x \ne x_0`` and ``V(x_0) = 0``.
We call such functions positive definite (around the fixed point ``x_0``).

Two structures are provided which partially or fully enforce the minimization condition: [`NonnegativeStructure`](@ref), which structurally enforces ``V(x) \ge 0`` everywhere, and [`PositiveSemiDefiniteStructure`](@ref), which additionally enforces ``V(x_0) = 0``.

```@docs
NonnegativeStructure
PositiveSemiDefiniteStructure
```

## Pre-defined Lux structures

Regardless of what NeuralLyapunov transformation is used to transform ``\phi`` into ``V``, users should carefully consider their choice of ``\phi``.
Two options provided by NeuralLyapunov, intended to be used with [`NoAdditionalStructure`](@ref), are [`AdditiveLyapunovNet`](@ref) and [`MultiplicativeLyapunovNet`](@ref).
These each wrap a different Lux model, effectively performing the transformation from ``\phi`` to ``V`` within the Lux ecosystem, rather than in the NeuralPDE/ModelingToolkit symbolic ecosystem. 

[`AdditiveLyapunovNet`](@ref) is based on [gaby_lyapunov-net_2021](@cite), and [`MultiplicativeLyapunovNet`](@ref) is an analogous structure combining the neural term and the positive definite term via multiplication instead of addition.

```@docs
AdditiveLyapunovNet
MultiplicativeLyapunovNet
```

Note that using [`NoAdditionalStructure`](@ref) with [`MultiplicativeLyapunovNet`](@ref) wrapping a Lux model ``\phi`` is the same as using [`PositiveSemiDefiniteStructure`](@ref) the same ``\phi``, but in the former the transformation is handled in the Lux ecosystem and in the latter the transformation is handled in the NeuralPDE/ModelingToolkit ecosystem.
Similarly, using [`NonnegativeStructure`](@ref) with [`Boltz.Layers.ShiftTo`](https://luxdl.github.io/Boltz.jl/dev/api/layers#Boltz.Layers-API-Reference) is analogous to using [`NoAdditionalStructure`](@ref) with [`AdditiveLyapunovNet`](@ref).
Because the NeuralPDE parser cannot process ``\phi`` being evaluated at two different points (in this case ``x`` and ``x_0``), we cannot represent this structure purely in the NeuralPDE/ModelingToolkit ecosystem.

Helper layers provided for the above structures are also exported:

```@docs
SoSPooling
StrictlyPositiveSoSPooling
```

## Defining your own neural Lyapunov function structure with [`NeuralLyapunovStructure`](@ref)

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

## References
```@bibliography
Pages = ["structure.md"]
```