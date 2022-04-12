# Bender.jl

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://rasmuskh.github.io/Bender.jl/dev/)
[![Build Status](https://github.com/Rasmuskh/Bender.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Rasmuskh/Bender.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/Rasmuskh/Bender.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Rasmuskh/Bender.jl)

A wide range of research on feedforward neural networks requires "bending" the chain rule during backpropagation. This package provides neural network layers (compatible with Flux.jl), which gives users more freedom to choose every aspect of the forward mapping. This makes it easy to leverage ChainRules.jl to compose a wide range of experiments, such as training binary neural networks, Feedback Alignment and Direct Feedback Alignment in just a few lines of code.

This package is not yet officially released so use at your own risk. Documentation and tests are work in progress, and API might change slightly before official release.

To install: enter package manager mode by typing `]` in a Julia REPL. Then type `add https://github.com/Rasmuskh/Bender.jl.git`.
