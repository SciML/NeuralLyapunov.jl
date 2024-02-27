# NeuralLyapunov.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://SciML.github.io/NeuralLyapunov.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://SciML.github.io/NeuralLyapunov.jl/dev/)
[![Build Status](https://github.com/SciML/NeuralLyapunov.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/SciML/NeuralLyapunov.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/SciML/NeuralLyapunov.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/SciML/NeuralLyapunov.jl)

A library for searching for neural Lyapunov functions in Julia.

This package provides an API for setting up the search for a neural Lyapunov function.
Such a search can be formulated as a partial differential inequality, and this library generates a ModelingToolkit PDESystem to be solved using NeuralPDE.jl.
Since the Lyapunov conditions can be formulated in several different ways and a neural Lyapunov function can be set up in many different forms, this library presents an extensible interface for users to choose how they wish to set up the search, with useful pre-built options for common setups.
