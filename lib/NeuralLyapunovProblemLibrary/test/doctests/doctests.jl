using Test, Documenter, NeuralLyapunovProblemLibrary

println("Running doctests for NeuralLyapunovProblemLibrary")

DocMeta.setdocmeta!(
    NeuralLyapunovProblemLibrary,
    :DocTestSetup,
    :(using NeuralLyapunovProblemLibrary, ModelingToolkit);
    recursive = true
)

doctest(NeuralLyapunovProblemLibrary; manual = false)
