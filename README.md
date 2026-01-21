# NeuralLyapunov.jl

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/NeuralLyapunov/stable/)

[![codecov](https://codecov.io/gh/SciML/NeuralLyapunov.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/SciML/NeuralLyapunov.jl)
[![Build Status](https://github.com/SciML/NeuralLyapunov.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/SciML/NeuralLyapunov.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Build status](https://badge.buildkite.com/201fa9f55f9b9f77b4a9e0cd6835e5a52ddbe7bc7fd7b724d3.svg)](https://buildkite.com/julialang/neurallyapunov-dot-jl)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

A library for searching for neural Lyapunov functions in Julia.

For information on using the package,
[see the stable documentation](https://docs.sciml.ai/NeuralLyapunov/stable/). Use the
[in-development documentation](https://docs.sciml.ai/NeuralLyapunov/dev/) for the version of
the documentation which contains the unreleased features.

## Overview

This package provides an API for setting up the search for a neural Lyapunov function.
Such a search can be formulated as a partial differential inequality, and this library generates a [ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/) PDESystem to be solved using [NeuralPDE.jl](https://docs.sciml.ai/NeuralPDE/stable/).
Since the Lyapunov conditions can be formulated in several different ways and a neural Lyapunov function can be set up in many different forms, this library presents an extensible interface for users to choose how they wish to set up the search, with useful pre-built options for common setups.
