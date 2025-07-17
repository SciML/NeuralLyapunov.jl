using NeuralPDE, NeuralLyapunov
import Optimization, OptimizationOptimisers, OptimizationOptimJL
using Random
using Lux, LuxCUDA, ComponentArrays
using Boltz.Layers: ShiftTo, MLP
using Test, LinearAlgebra, StableRNGs

rng = StableRNG(0)
Random.seed!(200)

println("Benchmark: Damped Simple Harmonic Oscillator (CUDA)")

######################### Define dynamics and domain ##########################

"Simple Harmonic Oscillator Dynamics"
function f(state, p, t)
    pos = state[1]
    vel = state[2]
    vcat(vel, -vel - pos)
end
lb = [-2.0, -2.0];
ub = [2.0, 2.0];
fixed_point = [0.0, 0.0];
dynamics = ODEFunction(f; sys = SciMLBase.SymbolCache([:x, :v]))

####################### Specify neural Lyapunov problem #######################

# Define neural network discretization
dim_state = length(lb)
dim_hidden = 20
chain = AdditiveLyapunovNet(
    MLP(dim_state, (dim_hidden, dim_hidden, dim_hidden, 1), tanh);
    dim_ϕ = 1,
    fixed_point
)
const gpud = gpu_device()
ps, st = Lux.setup(rng, chain)
ps = ps |> ComponentArray |> gpud |> f32
st = st |> gpud |> f32

# Define training strategy
strategy = QuasiRandomTraining(2500)
discretization = PhysicsInformedNN(chain, strategy; init_params = ps, init_states = st)

# Define neural Lyapunov structure
structure = NoAdditionalStructure()
minimization_condition = DontCheckNonnegativity()

# Define Lyapunov decrease condition
# This damped SHO has exponential decrease at a rate of k = 0.5, so we train to certify that
decrease_condition = ExponentialStability(0.5)

# Construct neural Lyapunov specification
spec = NeuralLyapunovSpecification(
    structure,
    minimization_condition,
    decrease_condition
)

############################ Try Benchmarking ############################
# Define optimization parameters
opt = [
    OptimizationOptimisers.Adam(0.01),
    OptimizationOptimisers.Adam(),
    OptimizationOptimJL.BFGS()
]
optimization_args = [:maxiters => 300]

out = benchmark(
    dynamics,
    lb,
    ub,
    spec,
    chain,
    strategy,
    opt;
    fixed_point,
    simulation_time = 300,
    n = 1000,
    optimization_args,
    rng,
    init_params = ps,
    init_states = st
)

# AdditiveLyapunovNet should have no trouble with the globally stable damped SHO, so we
# expect it to correctly classify everything as within the region of attraction.
@test out.confusion_matrix.p + out.confusion_matrix.n == out.confusion_matrix.tp
