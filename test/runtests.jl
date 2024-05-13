using SafeTestsets

@time begin
    @time @safetestset "Damped simple harmonic oscillator" begin
        include("damped_sho.jl")
    end
    @time @safetestset "Damped pendulum" begin
        include("damped_pendulum.jl")
    end
    @time @safetestset "Region of attraction estimation" begin
        include("roa_estimation.jl")
    end
    @time @safetestset "Policy search - inverted pendulum" begin
        include("inverted_pendulum.jl")
    end
    @time @safetestset "Policy search - inverted pendulum 2" begin
        include("inverted_pendulum_ODESystem.jl")
    end
end
