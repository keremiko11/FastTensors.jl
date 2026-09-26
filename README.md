# 📐 FastTensors.jl

High-performance multidimensional tensor algebra and contractions written in idiomatic Julia, featuring cache-aware loop tiling, SIMD vectorization, and Higher-Order SVD (HOSVD) decompositions.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Julia](https://img.shields.io/badge/Julia-1.9%2B-9558B2.svg)](https://julialang.org)
[![Pkg](https://img.shields.io/badge/Pkg.jl-v0.2.0-blueviolet.svg)](https://github.com/keremiko11/FastTensors.jl)

## Features

- ⚡ **Cache-Oblivious Tiling** — Optimized contraction algorithms maximizing L1/L2 cache locality.
- 🚀 **SIMD & Loop Vectorization** — Native AVX2/AVX-512 vectorization via Julia's LLVM backend.
- 🧮 **Higher-Order SVD (Tucker & CP)** — Tensor decomposition tools for low-rank approximation.
- 🛡️ **Zero-Cost Abstractions** — Contiguous StridedViews with non-allocating slicing and reshapes.
- 📊 **Type-Stable** — Strict parametric signatures guaranteeing compile-time specialization.

## Installation

Within the Julia REPL:

```julia
using Pkg
Pkg.add(url="https://github.com/keremiko11/FastTensors.jl.git")
```

## Quick Start

```julia
using FastTensors

# Create a 3D tensor from dimensions
T = rand_tensor(Float64, (64, 64, 32))

# Compute tensor Frobenius norm
println("Frobenius Norm: ", tensor_norm(T))

# High-performance tensor contraction along matching indices
A = rand_tensor(Float64, (32, 16, 8))
B = rand_tensor(Float64, (8, 16, 24))

# Contract along index 2 and 3
C = contract(A, B, (2, 3), (2, 1))
println("Contracted shape: ", size(C))

# Truncated Higher-Order SVD (Tucker decomposition)
core, factors = hosvd(T, ranks=(16, 16, 8))
println("Compressed core shape: ", size(core))
```

## Running Tests

Launch Julia test runner:

```julia
using Pkg
Pkg.test("FastTensors")
```

## License

MIT License - Copyright (c) 2024 Kerem Koç.
