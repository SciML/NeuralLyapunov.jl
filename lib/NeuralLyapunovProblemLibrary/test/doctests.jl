using Test, Documenter, NeuralLyapunovProblemLibrary

DocMeta.setdocmeta!(
    NeuralLyapunovProblemLibrary,
    :DocTestSetup,
    :(
        using NeuralLyapunovProblemLibrary, ModelingToolkit, OrdinaryDiffEq, Random;
        Random.seed!(200)
    );
    recursive = true
)

doctest(NeuralLyapunovProblemLibrary; manual = false)
