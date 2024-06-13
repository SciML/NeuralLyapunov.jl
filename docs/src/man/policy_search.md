# Policy Search and Network-Sependent Dynamics

At times, we wish to model a component of the dynamics with a neural network.
A common example is the policy search case, when the closed-loop dynamics include a neural network controller.
In such cases, we consider the dynamics to take the form of ``\frac{dx}{dt} = f(x, u, p, t)``, where ``u`` is the control input/the contribution to the dynamics from the neural network.
We provide the [`add_policy_search`](@ref) function to transform a [`NeuralLyapunovStructure`](@ref) to include training the neural network to represent not just the Lyapunov function, but also the relevant part of the dynamics.

Similar to [`get_numerical_lyapunov_function`](@ref), we provide the [`get_policy`](@ref) convenience function to construct ``u(x)`` that can be combined with the open-loop dynamics ``f(x, u, p, t)`` to create closed loop dynamics ``f_{cl}(x, p, t) = f(x, u(x), p, t)``.

```@docs
add_policy_search
get_policy
```
