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
end
