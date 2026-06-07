using Pkg
using SafeTestsets: @safetestset

const GROUP_RAW = get(ENV, "GROUP", "all")
const GROUP = lowercase(GROUP_RAW)
const DEVICE = lowercase(get(ENV, "DEVICE", "cpu"))

# Detect sublibrary test groups.
# GROUP can be a bare sublibrary name (Core test group) or
# "{sublibrary}_{TEST_GROUP}" for any custom group (e.g., qa, pendula, ...).
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

const BASE_GROUP, TEST_GROUP = _detect_sublibrary_group(GROUP_RAW, LIB_DIR)

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
        if GROUP == "all" || GROUP == "core"
            if DEVICE == "cpu"
                @time @safetestset "Damped simple harmonic oscillator" begin
                    include("damped_sho.jl")
                end
                @time @safetestset "Damped pendulum" begin
                    include("damped_pendulum.jl")
                end
                @time @safetestset "Damped pendulum - AdditiveLyapunovNet structure" begin
                    include("damped_pendulum_lux.jl")
                end
                @time @safetestset "Damped pendulum - MultiplicativeLyapunovNet structure" begin
                    include("damped_pendulum_lux_2.jl")
                end
            end

            if DEVICE == "gpu"
                @time @safetestset "CUDA test - Damped SHO" begin
                    include("damped_sho_CUDA.jl")
                end
            end
        end

        if GROUP == "all" || GROUP == "policy_search"
            if DEVICE == "cpu"
                @time @safetestset "Policy search - inverted pendulum" begin
                    include("inverted_pendulum.jl")
                end
                @time @safetestset "Policy search - inverted pendulum (ODESystem)" begin
                    include("inverted_pendulum_ODESystem.jl")
                end
            end
        end

        if GROUP == "all" || GROUP == "roa"
            if DEVICE == "cpu"
                @time @safetestset "Region of attraction estimation" begin
                    include("roa_estimation.jl")
                end
            end
        end

        if GROUP == "all" || GROUP == "local_lyapunov"
            if DEVICE == "cpu"
                @time @safetestset "Local Lyapunov function search" begin
                    include("local_lyapunov.jl")
                end
            end
        end

        if GROUP == "all" || GROUP == "benchmarking"
            if DEVICE == "cpu"
                @time @safetestset "Benchmarking tool" begin
                    include("benchmark.jl")
                end
            end
            if DEVICE == "gpu"
                @time @safetestset "Benchmarking tool - CUDA" begin
                    include("benchmark_CUDA.jl")
                end
            end
        end

        if GROUP == "all" || GROUP == "unimplemented"
            if DEVICE == "cpu"
                @time @safetestset "Errors for partially-implemented extensions" begin
                    include("unimplemented.jl")
                end
            end
        end

        if GROUP == "all" || GROUP == "qa"
            if DEVICE == "cpu"
                @time @safetestset "Quality Assurance" begin
                    include("qa_tests.jl")
                end
                #= JET tests are not essential and currently terminate unexpectedly
                @time @safetestset "JET: Static Analysis" begin
                    include("jet_tests.jl")
                end
                =#
            end
        end

        if GROUP == "all" || GROUP == "doctests"
            if DEVICE == "cpu"
                @time @safetestset "Doctests" begin
                    include("doctests.jl")
                end
            end
        end
    end
end
