# Developer API

This page documents the developer-facing extension points used by
`NeuralLyapunovPDESystem`. These interfaces are version-controlled and tested for
compatibility with the generic package code, but they are intended for packages that build
new NeuralLyapunov formulations rather than for ordinary application code. Implement a
subtype and extend the documented generic functions; do not depend on fields of the built-in
concrete types or on names omitted from this page.

```@docs
NeuralLyapunov.NeuralLyapunov
NeuralLyapunov.phi_to_net
NeuralLyapunov.NeuralLyapunovBenchmarkLogger
NeuralLyapunov.AbstractNeuralLyapunovStructure
NeuralLyapunov.get_V
NeuralLyapunov.get_V̇
NeuralLyapunov.neural_controller
NeuralLyapunov.get_network_dim
NeuralLyapunov.get_control_dim
NeuralLyapunov.get_control_structure
```
