using SciMLTesting, NeuralLyapunovProblemLibrary, Test

run_qa(
    NeuralLyapunovProblemLibrary;
    explicit_imports = true,
    ei_kwargs = (;
        no_implicit_imports = (; skip = (Base, Core)),
    ),
)
