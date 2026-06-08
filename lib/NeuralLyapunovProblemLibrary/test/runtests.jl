using Pkg
using SafeTestsets: @safetestset

const GROUP = get(ENV, "NEURALLYAPUNOV_TEST_GROUP", get(ENV, "GROUP", "All"))

# The QA group adds Aqua + ExplicitImports beyond the package's base test deps,
# so it carries its own test/qa/Project.toml and is excluded from the `All` run
# (which executes the base groups in the package's main test env).
function activate_qa_env()
    Pkg.activate(joinpath(@__DIR__, "qa"))
    return Pkg.instantiate()
end

@time begin
    if GROUP == "All" || GROUP == "Core" || GROUP == "Pendula"
        @time @safetestset "Simple pendulum" begin
            include("pendula/pendulum_test.jl")
        end

        @time @safetestset "Double pendulum" begin
            include("pendula/double_pendulum_test.jl")
        end
    end

    if GROUP == "All" || GROUP == "Core" || GROUP == "Quadrotors"
        @time @safetestset "Planar Quadrotor" begin
            include("quadrotors/planar_quadrotor_test.jl")
        end

        @time @safetestset "Quadrotor (3D)" begin
            include("quadrotors/quadrotor_test.jl")
        end
    end

    if GROUP == "All" || GROUP == "Doctests"
        @time @safetestset "Doctests" begin
            include("doctests/doctests.jl")
        end
    end

    # QA is a dep-adding group: it activates its own env and is excluded from `All`.
    if GROUP == "QA"
        activate_qa_env()
        @time @safetestset "Quality Assurance" begin
            include("qa/qa.jl")
        end
        @time @safetestset "Explicit Imports" begin
            include("qa/explicit_imports.jl")
        end
    end
end
