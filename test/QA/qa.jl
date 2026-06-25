using SciMLTesting, NeuralLyapunov, Test

# We need a Symbolics internal (diff2term) and for some reason QuasiMonteCarlo doesn't
# export sample or mark it as public
if VERSION >= v"1.11.0-DEV.469"
    explicit_imports_public_ignore = (:diff2term, :sample)
else
    # ShiftTo, checksquare, and unbound_inputs are public in Julia v1.11+, but not
    # exported in Julia v1.10
    explicit_imports_public_ignore = (:diff2term, :sample, :ShiftTo, :checksquare,
        :unbound_inputs)
end

# ForwardDiff doesn't export derivative, gradient, or jacobian, nor does SciMLBase
# export NullParameters, __has_jac, or __has_control_jac, nor does Symbolics export
# value
if VERSION >= v"1.11.0-DEV.469"
    qualified_accesses_public_ignore = (
        :NullParameters, :derivative, :gradient, :jacobian, :logscalar, :__has_jac,
        :__has_controljac, :value,
    )
else
    # Fix1, Fix2, initialparameters, and initialstates are public in Julia v1.11+, but
    # not exported in Julia v1.10
    qualified_accesses_public_ignore = (
        :NullParameters, :derivative, :gradient, :jacobian, :logscalar, :__has_jac,
        :__has_controljac, :value, :Fix1, :Fix2, :initialparameters, :initialstates,
    )
end

run_qa(
    NeuralLyapunov;
    explicit_imports = true,
    ei_kwargs = (;
        no_implicit_imports = (; skip = (Base, Core)),
        all_explicit_imports_are_public = (; ignore = explicit_imports_public_ignore),
        all_qualified_accesses_are_public = (; ignore = qualified_accesses_public_ignore),
    ),
)
