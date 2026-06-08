using Test, Documenter, NeuralLyapunov

println("Running doctests")

DocMeta.setdocmeta!(
    NeuralLyapunov,
    :DocTestSetup,
    :(using NeuralLyapunov);
    recursive = true
)

doctest(NeuralLyapunov; manual = false)
