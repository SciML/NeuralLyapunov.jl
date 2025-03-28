# Benchmarking neural Lyapunov methods

To facilitate comparison of different neural Lyapunov specifications, optimizers, hyperparameters, etc., we provide the [`benchmark`](@ref) function.

Through its arguments, users may specify how a neural Lyapunov problem, the neural network structure, the physics-informed neural network discretization strategy, and the optimization strategy used to solve the problem.
After solving the problem in the specified manner, the dynamical system is simulated (users can specify an ODE solver in the arguments, as well) and classification by the neural Lyapunov function is compared to the simulation results.
The [`benchmark`](@ref) function returns a confusion matrix for the resultant neural Lyapunov classifier, the training time, and samples with labels, so that users can compare accuracy and computation speed of various methods.

```@docs
benchmark
```