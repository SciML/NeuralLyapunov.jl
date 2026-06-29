using SciMLTesting, NeuralLyapunov, Test

# Symbolics doesn't mark diff2term as public, nor does QuasiMonteCarlo export or mark
# sample as public.
explicit_imports_public_ignore = (:diff2term, :sample)

# ForwardDiff doesn't export or mark derivative, gradient, or jacobian as public, nor
# does NeuralPDE mark logscalar as public.
qualified_accesses_public_ignore = (:derivative, :gradient, :jacobian, :logscalar)

run_qa(
    NeuralLyapunov;
    explicit_imports = true,
    ei_kwargs = (;
        no_implicit_imports = (; skip = (Base, Core)),
        all_explicit_imports_are_public = (; ignore = explicit_imports_public_ignore),
        all_qualified_accesses_are_public = (; ignore = qualified_accesses_public_ignore),
    ),
)
