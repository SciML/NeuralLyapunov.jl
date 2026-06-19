using ExplicitImports
using Test: @test
import NeuralLyapunovProblemLibrary

@testset "Explicit Imports" begin
    @test check_no_implicit_imports(NeuralLyapunovProblemLibrary; skip = (Base, Core)) ===
        nothing
    @test check_no_stale_explicit_imports(NeuralLyapunovProblemLibrary) === nothing
    @test check_no_self_qualified_accesses(NeuralLyapunovProblemLibrary) === nothing
    @test check_all_explicit_imports_via_owners(NeuralLyapunovProblemLibrary) === nothing
    @test check_all_qualified_accesses_via_owners(NeuralLyapunovProblemLibrary) === nothing
    @test check_all_qualified_accesses_are_public(NeuralLyapunovProblemLibrary) === nothing

    # SciMLBase doesn't export NullParameters
    if VERSION >= v"1.11.0-DEV.469"
        ignore = (:NullParameters)
    else
        # unbound_inputs is public in Julia v1.11+, but not exported in Julia v1.10
        ignore = (:NullParameters, :unbound_inputs)
    end
    @test check_all_explicit_imports_are_public(NeuralLyapunovProblemLibrary; ignore) ===
        nothing

end
