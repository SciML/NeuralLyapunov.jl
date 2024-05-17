# Solving a Neural Lyapunov Problem

NeuralLyapunov.jl represents neural Lyapunov problems as systems of partial differential equations, using the `ModelingToolkit.PDESystem` type.
Such a `PDESystem` can then be solved using a physics-informed neural network through [NeuralPDE.jl](https://github.com/SciML/NeuralPDE.jl).

Candidate Lyapunov functions will be trained within a box domain subset of the state space.

```@docs
NeuralLyapunovPDESystem
```

## Extracting the numerical Lyapunov function

We provide the following convenience function for generating the Lyapunov function after the parameters have been found.
If the `PDESystem` was solved using `NeuralPDE.jl` and `Optimization.jl`, then the argument `phi` is a field of the output of `NeuralPDE.discretize` and the argument `θ` is `res.u.depvar` where `res` is the result of the optimization.

```@docs
get_numerical_lyapunov_function
```
