# Max-Tree GPU Benchmark Suite

## Introduction

This repository contains code for running **benchmark experiments** adapted from the following project:

> https://gitlab.lre.epita.fr/nblin/max-tree

The original codebase was used in the paper:

> **Nicolas Blin, Edwin Carlinet, Florian Lemaitre, Lionel Lacassagne, Thierry Géraud**  
> *Max-tree Computation on GPUs*, 2022. ⟨hal-03556296⟩

The implementation provides multiple **CPU and GPU variants** of the max-tree construction algorithm, including several GPU-specific optimizations.

⚠️ **Important note**  
This code is intended **only for benchmarking and experimental reproduction**. It does **not** expose a stable or user-friendly API and should not be used as a general-purpose library.

---

## Build Instructions

### Prerequisites

- CMake
- Conan (v2)
- CUDA Toolkit (tested with CUDA ≥ 12)
- Ninja (recommended)

Ensure that `nvcc` is correctly installed and note its absolute path.

---

### Build Steps

From the root of the repository:

```shell
> mkdir build && cd build
> conan install .. --output-folder=. --build=missing -s compiler.cppstd=20
> cmake .. \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12/bin/nvcc \
    -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -G Ninja
> ninja
```

## Running the Benchmarks

All benchmark executables are generated in the `bin/` directory after a successful build.

Before running any benchmark, make sure you have downloaded the test image (see **Test Image** section below) and that you are running from the project root or `build/` directory as appropriate.

---

### Benchmark Modes

The benchmark supports two execution modes:

---

### 1. Full pipeline benchmark (with memory transfers)

This mode measures **end-to-end performance**, including:

- Host → Device memory transfers
- GPU kernel execution
- Device → Host memory transfers

Run:

```shell
> ./bin/TestMaxtree --no-check ./test.pgm
```

### 2. Kernel-only benchmark (no memory transfers)

This mode measures **GPU kernel execution time only**, excluding all host–device memory transfers. It is useful for evaluating the intrinsic efficiency of the CUDA kernels without PCIe overhead.

Run:

```shell
> ./bin/TestMaxtree --no-check --bench-kernel-only ./test.pgm
```

## Test Images

The benchmark experiments use grayscale images in **PGM format**.

The primary test image used in the experiments can be downloaded from the following link:

- **Test image (PGM)**:  
  https://drive.google.com/file/d/1kb79XAXE1A_ioj8cBvhV2H70QLDZDSXg/view?usp=drive_link


