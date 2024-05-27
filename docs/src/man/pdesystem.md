# Solving a Neural Lyapunov Problem

NeuralLyapunov.jl represents neural Lyapunov problems as systems of partial differential equations, using the `ModelingToolkit.PDESystem` type.
Such a `PDESystem` can then be solved using a physics-informed neural network through [NeuralPDE.jl](https://github.com/SciML/NeuralPDE.jl).

Candidate Lyapunov functions will be trained within a box domain subset of the state space.

```@docs
NeuralLyapunovPDESystem
```

!!! warning

    When using [`NeuralLyapunovPDESystem`](@ref), the Lyapuonv function structure, minimization and decrease conditions, and dynamics will all be symbolically traced to generate the resulting `PDESystem` equations.
    In some cases tracing results in more efficient code, but in others it can result in inefficiencies or even errors.
    
    If the generated `PDESystem` is then used with NeuralPDE.jl, that library's parser will convert the equations into Julia functions representing the loss, which presents another opportunity for unexpected errors.

## Extracting the numerical Lyapunov function

We provide the following convenience function for generating the Lyapunov function after the parameters have been found.
If the `PDESystem` was solved using NeuralPDE.jl and Optimization.jl, then the argument `phi` is a field of the output of `NeuralPDE.discretize` and the argument `θ` is `res.u.depvar` where `res` is the result of the optimization.

```@docs
get_numerical_lyapunov_function
```
