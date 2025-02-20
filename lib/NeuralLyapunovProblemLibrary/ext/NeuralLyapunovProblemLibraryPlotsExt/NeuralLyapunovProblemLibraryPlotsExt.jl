module NeuralLyapunovProblemLibraryPlotsExt

using NeuralLyapunovProblemLibrary
import NeuralLyapunovProblemLibrary: plot_pendulum
using Plots

include("pendulum_plot.jl")
include("double_pendulum_plot.jl")

end
