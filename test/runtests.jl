using Pkg
using SafeTestsets: @safetestset

const GROUP = get(ENV, "GROUP", "All")

# Detect sublibrary test groups.
# GROUP can be a bare sublibrary name (Core test group) or
# "{sublibrary}_{TEST_GROUP}" for any custom group (e.g., QA, Pendula, ...).
# Sublibraries declare their groups in test/test_groups.toml and read the group
# from NEURALLYAPUNOV_TEST_GROUP.
const LIB_DIR = joinpath(@__DIR__, "..", "lib")

function _detect_sublibrary_group(group, lib_dir)
    isdir(joinpath(lib_dir, group)) && return (group, "Core")
    for i in length(group):-1:1
        if group[i] == '_' && isdir(joinpath(lib_dir, group[1:(i - 1)]))
            return (group[1:(i - 1)], group[(i + 1):end])
        end
    end
    return (group, "Core")
end

const BASE_GROUP, TEST_GROUP = _detect_sublibrary_group(GROUP, LIB_DIR)

# Dep-adding groups carry their own test/<group>/Project.toml; activate +
# instantiate it before including that group's tests. These groups are NOT part
# of the `All` run, which executes only the base groups in the main test env.
function activate_group_env(group)
    Pkg.activate(joinpath(@__DIR__, group))
    return Pkg.instantiate()
end

@time begin
    if isdir(joinpath(LIB_DIR, BASE_GROUP))
        Pkg.activate(joinpath(LIB_DIR, BASE_GROUP))
        # On Julia < 1.11 the [sources] section is unsupported; develop the
        # sublibrary's in-repo path dependencies (transitively) so CI tests the
        # PR branch code. NeuralLyapunov's floor is 1.11 so this is effectively
        # a no-op today, but mirrors the canonical dispatcher.
        if VERSION < v"1.11.0-DEV.0"
            developed = Set{String}()
            push!(developed, normpath(joinpath(LIB_DIR, BASE_GROUP)))
            specs = Pkg.PackageSpec[]
            queue = [joinpath(LIB_DIR, BASE_GROUP)]
            while !isempty(queue)
                pkg_dir = popfirst!(queue)
                toml_path = joinpath(pkg_dir, "Project.toml")
                isfile(toml_path) || continue
                toml = Pkg.TOML.parsefile(toml_path)
                if haskey(toml, "sources")
                    for (dep_name, source_spec) in toml["sources"]
                        if source_spec isa Dict && haskey(source_spec, "path")
                            dep_path = normpath(joinpath(pkg_dir, source_spec["path"]))
                            if isdir(dep_path) && !(dep_path in developed)
                                push!(developed, dep_path)
                                push!(specs, Pkg.PackageSpec(path = dep_path))
                                push!(queue, dep_path)
                            end
                        end
                    end
                end
            end
            isempty(specs) || Pkg.develop(specs)
        end
        withenv("NEURALLYAPUNOV_TEST_GROUP" => TEST_GROUP) do
            Pkg.test(BASE_GROUP, force_latest_compatible_version = false, allow_reresolve = true)
        end
    else
        # Base groups: run in the main test env and are part of `All`.
        if GROUP == "All" || GROUP == "Core"
            @time @safetestset "Damped simple harmonic oscillator" begin
                include("core/damped_sho.jl")
            end
            @time @safetestset "Damped pendulum" begin
                include("core/damped_pendulum.jl")
            end
            @time @safetestset "Damped pendulum - AdditiveLyapunovNet structure" begin
                include("core/damped_pendulum_lux.jl")
            end
            @time @safetestset "Damped pendulum - MultiplicativeLyapunovNet structure" begin
                include("core/damped_pendulum_lux_2.jl")
            end
        end

        if GROUP == "All" || GROUP == "Policy_search"
            @time @safetestset "Policy search - inverted pendulum" begin
                include("policy_search/inverted_pendulum.jl")
            end
            @time @safetestset "Policy search - inverted pendulum (ODESystem)" begin
                include("policy_search/inverted_pendulum_ODESystem.jl")
            end
        end

        if GROUP == "All" || GROUP == "ROA"
            @time @safetestset "Region of attraction estimation" begin
                include("roa/roa_estimation.jl")
            end
        end

        if GROUP == "All" || GROUP == "Local_lyapunov"
            @time @safetestset "Local Lyapunov function search" begin
                include("local_lyapunov/local_lyapunov.jl")
            end
        end

        if GROUP == "All" || GROUP == "Benchmarking"
            @time @safetestset "Benchmarking tool" begin
                include("benchmarking/benchmark.jl")
            end
        end

        if GROUP == "All" || GROUP == "Unimplemented"
            @time @safetestset "Errors for partially-implemented extensions" begin
                include("unimplemented/unimplemented.jl")
            end
        end

        # Dep-adding groups: each carries its own Project.toml and is excluded
        # from `All` (which runs in the main test env).
        if GROUP == "QA"
            activate_group_env("qa")
            @time @safetestset "Quality Assurance" begin
                include("qa/qa.jl")
            end
            @time @safetestset "Explicit Imports" begin
                include("qa/explicit_imports.jl")
            end
            #= JET tests are not essential and currently terminate unexpectedly
            @time @safetestset "JET: Static Analysis" begin
                include("qa/jet.jl")
            end
            =#
        end

        if GROUP == "Doctests"
            activate_group_env("doctests")
            @time @safetestset "Doctests" begin
                include("doctests/doctests.jl")
            end
        end

        if GROUP == "GPU"
            activate_group_env("gpu")
            @time @safetestset "CUDA test - Damped SHO" begin
                include("gpu/damped_sho_CUDA.jl")
            end
            @time @safetestset "Benchmarking tool - CUDA" begin
                include("gpu/benchmark_CUDA.jl")
            end
        end
    end
end
