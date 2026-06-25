using SciMLTesting, NeuralLyapunovProblemLibrary, Test

# SciMLBase doesn't export NullParameters
if VERSION >= v"1.11.0-DEV.469"
    explicit_imports_public_ignore = (:NullParameters,)
else
    # unbound_inputs is public in Julia v1.11+, but not exported in Julia v1.10
    explicit_imports_public_ignore = (:NullParameters, :unbound_inputs)
end

run_qa(
    NeuralLyapunovProblemLibrary;
    explicit_imports = true,
    ei_kwargs = (;
        no_implicit_imports = (; skip = (Base, Core)),
        all_explicit_imports_are_public = (; ignore = explicit_imports_public_ignore),
    ),
)
