using Test, NeuralLyapunov
using JET: report_package, get_reports, @test_opt

@testset "JET Static Analysis" begin

    # Test key exported functions for type stability
    # Note: We use report_package with target_modules to analyze just NeuralLyapunov code
    @testset "Package-level JET analysis" begin
        result = report_package(
            NeuralLyapunov;
            target_modules = (NeuralLyapunov,)
        )
        reports = get_reports(result)
        # Filter out JuMP-related reports which are false positives from JuMP's macro system
        # These are issues in JuMP's @variable macro, not in NeuralLyapunov code
        relevant_reports = filter(reports) do r
            !occursin("JuMP", string(r)) && !occursin("build_variable", string(r))
        end
        @test length(relevant_reports) == 0
    end

    # Test specific key functions for optimization opportunities
    @testset "Lyapunov conditions" begin
        # Test that condition constructors are type-stable
        @test_opt StrictlyPositiveDefinite()
        @test_opt PositiveSemiDefinite()
        @test_opt DontCheckNonnegativity()
        @test_opt AsymptoticStability()
        @test_opt ExponentialStability(1.0)
        @test_opt DontCheckDecrease()
        @test_opt StabilityISL()
    end

    @testset "Structure functions" begin
        @test_opt NoAdditionalStructure()
        @test_opt NonnegativeStructure(2)
        @test_opt PositiveSemiDefiniteStructure(2)
    end
end
