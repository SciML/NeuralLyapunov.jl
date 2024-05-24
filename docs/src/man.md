# Components of a Neural Lyapunov Problem

For a candidate Lyapunov function $V(x)$ to certify the stability of an equilibrium point $x_0$ of the dynamical system $\frac{dx}{dt} = f(x(t))$, it must satisfy two conditions:
1. The function $V$ must be uniquely minimized at $x_0$, and 
2. The function $V$ must decrease along system trajectories (i.e., $V(x(t))$ decreases as long as $x(t)$ is a trajectory of the dynamical system).

A neural Lyapunov function represents the candidate Lyapunov function $V$ using a neural network, sometimes modifying the output of the network slightly so as to enforce one of the above conditions.

Thus, we specify our neural Lyapunov problems with three components, each answering a different question:
1. How is $V$ structured in terms of the neural network?
2. How is the minimization condition to be enforced?
3. How is the decrease condition to be enforced?

These three components are represented by the three fields of a `NeuralLyapunovSpecification` object.

```@docs
NeuralLyapunovSpecification
```
