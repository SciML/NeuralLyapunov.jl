using SafeTestsets: @safetestset

@safetestset "Aqua: Quality Assurance" begin
    println("Aqua quality assurance tests")
    import Aqua, NeuralLyapunovProblemLibrary, Test

    Aqua.test_all(NeuralLyapunovProblemLibrary)
end

@safetestset "Explicit Imports: Quality Assurance" begin
    println("Explicit imports tests")
    using ExplicitImports
    using Test: @test
    import NeuralLyapunovProblemLibrary

    @test check_no_implicit_imports(NeuralLyapunovProblemLibrary; skip = (Base, Core)) ===
        nothing
    @test check_no_stale_explicit_imports(NeuralLyapunovProblemLibrary) === nothing
    @test check_no_self_qualified_accesses(NeuralLyapunovProblemLibrary) === nothing
    @test check_all_explicit_imports_via_owners(NeuralLyapunovProblemLibrary) === nothing
    @test check_all_qualified_accesses_via_owners(NeuralLyapunovProblemLibrary) === nothing
    @test check_all_explicit_imports_are_public(
        NeuralLyapunovProblemLibrary; ignore = (:NullParameters, :unbound_inputs)
    ) === nothing
    @test check_all_qualified_accesses_are_public(NeuralLyapunovProblemLibrary) === nothing
end
