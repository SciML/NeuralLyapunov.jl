module NeuralLyapunovProblemLibraryPlotsExt

import NeuralLyapunovProblemLibrary
using NeuralLyapunovProblemLibrary: plot_pendulum, plot_double_pendulum,
    plot_quadrotor_planar, plot_quadrotor_3d
import Plots
using Plots: Shape, plot, plot!, quiver!, scatter!, annotate!, @animate, title!, xlims,
    xlims!, ylims, ylims!, zlims, zlims!
using Rotations: RotZXY

include("pendulum_plot.jl")
include("double_pendulum_plot.jl")
include("quadrotor_plot.jl")

end
