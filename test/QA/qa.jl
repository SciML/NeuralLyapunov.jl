using SciMLTesting, NeuralLyapunov, Test

run_qa(
    NeuralLyapunov;
    ei_kwargs = (; all_explicit_imports_are_public = (; ignore = (:Phi,)))
)
