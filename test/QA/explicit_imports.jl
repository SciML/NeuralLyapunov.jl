using ExplicitImports
using Test: @test
import NeuralLyapunov

@testset "Explicit Imports" begin
    @test check_no_implicit_imports(NeuralLyapunov; skip = (Base, Core)) === nothing
    @test check_no_stale_explicit_imports(NeuralLyapunov) === nothing
    @test check_no_self_qualified_accesses(NeuralLyapunov) === nothing
    @test check_all_explicit_imports_via_owners(NeuralLyapunov) === nothing
    @test check_all_qualified_accesses_via_owners(NeuralLyapunov) === nothing

    # We need a Symbolics internal (diff2term) and for some reason QuasiMonteCarlo doesn't
    # export sample or mark it as public
    if VERSION >= v"1.11.0-DEV.469"
        ignore = (:diff2term, :sample)
    else
        # ShiftTo, checksquare, and unbound_inputs are public in Julia v1.11+, but not
        # exported in Julia v1.10
        ignore = (:diff2term, :sample, :ShiftTo, :checksquare, :unbound_inputs)
    end
    @test check_all_explicit_imports_are_public(NeuralLyapunov; ignore) === nothing

    # ForwardDiff doesn't export derivative, gradient, or jacobian, nor does SciMLBase
    # export NullParameters, __has_jac, or __has_control_jac, nor does Symbolics export
    # value
    if VERSION >= v"1.11.0-DEV.469"
        ignore = (
            :NullParameters, :derivative, :gradient, :jacobian, :logscalar, :__has_jac,
            :__has_controljac, :value,
        )
    else
        # Fix1, Fix2, initialparameters, and initialstates are public in Julia v1.11+, but
        # not exported in Julia v1.10
        ignore = (
            :NullParameters, :derivative, :gradient, :jacobian, :logscalar, :__has_jac,
            :__has_controljac, :value, :Fix1, :Fix2, :initialparameters, :initialstates,
        )
    end
    @test check_all_qualified_accesses_are_public(NeuralLyapunov; ignore) === nothing
end
