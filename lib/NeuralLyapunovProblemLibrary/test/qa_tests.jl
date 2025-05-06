using Test, NeuralLyapunovProblemLibrary

@testset "Aqua: Quality Assurance" begin
    using Aqua

    Aqua.test_all(NeuralLyapunovProblemLibrary)
end

@testset "Explicit Imports: Quality Assurance" begin
    using ExplicitImports

    @test check_no_implicit_imports(NeuralLyapunovProblemLibrary; skip = (Base, Core)) ===
          nothing
    @test check_no_stale_explicit_imports(NeuralLyapunovProblemLibrary) === nothing
    @test check_no_self_qualified_accesses(NeuralLyapunovProblemLibrary) === nothing
    @test check_all_explicit_imports_via_owners(NeuralLyapunovProblemLibrary) === nothing
    @test check_all_qualified_accesses_via_owners(NeuralLyapunovProblemLibrary) === nothing
    @test check_all_explicit_imports_are_public(
        NeuralLyapunovProblemLibrary; ignore = (:NullParameters,)) === nothing
    @test check_all_qualified_accesses_are_public(NeuralLyapunovProblemLibrary) === nothing
end
