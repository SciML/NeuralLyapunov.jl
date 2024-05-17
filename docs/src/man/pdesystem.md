# Solving a Neural Lyapunov Problem

NeuralLyapunov.jl represents neural Lyapunov problems as systems of partial differential equations, using the `ModelingToolkit.PDESystem` type.
Such a `PDESystem` can then be solved using a physics-informed neural network through [NeuralPDE.jl](https://github.com/SciML/NeuralPDE.jl).

```@docs
NeuralLyapunovPDESystem
```

## Extracting the numerical Lyapunov function

```@docs
get_numerical_lyapunov_function
```
