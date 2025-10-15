using Test, NeuralLyapunov

@testset "Aqua: Quality Assurance" begin
    using Aqua

    Aqua.test_all(NeuralLyapunov)
end

@testset "Explicit Imports: Quality Assurance" begin
    using ExplicitImports

    @test check_no_implicit_imports(NeuralLyapunov; skip = (Base, Core)) === nothing
    @test check_no_stale_explicit_imports(NeuralLyapunov) === nothing
    @test check_no_self_qualified_accesses(NeuralLyapunov) === nothing
    @test check_all_explicit_imports_via_owners(NeuralLyapunov) === nothing
    @test check_all_qualified_accesses_via_owners(NeuralLyapunov) === nothing

    # We need a couple Symbolics internals (diff2term and value) and a ModelingToolkit
    # internal (unbound_inputs), and for some reason QuasiMonteCarlo doesn't export sample
    @test check_all_explicit_imports_are_public(
        NeuralLyapunov; ignore = (:diff2term, :value, :sample, :unbound_inputs)) === nothing

    # ForwardDiff doesn't export derivative, gradient, or jacobian, nor does SciMLBase with
    # NullParameters
    @test check_all_qualified_accesses_are_public(
        NeuralLyapunov;
        ignore = (:NullParameters, :derivative, :gradient, :jacobian, :logscalar)
    ) === nothing
end
