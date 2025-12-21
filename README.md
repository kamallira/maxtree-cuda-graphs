# Modernized CUDA Maxtree with Graph Optimization

This repository is a **revitalized and optimized** implementation of the Maxtree algorithm, originally developed by Nicolas Blin (https://gitlab.lre.epita.fr/nblin/max-tree). 

The original codebase had become incompatible with modern build environments due to outdated dependencies and deprecated libraries. This project restores functionality for modern systems and introduces significant performance improvements using **CUDA Graphs**.

### Key Improvements
* **Modernization:**  Updated build system (CMake/Conan) to support current CUDA toolkits.
    * Refactored code to resolve dependency conflicts and deprecated API calls.
* **Optimization:**  Implemented **CUDA Graphs** to replace iterative kernel launches, reducing CPU-side latency.

The **`maxtree_original/`** directory contains the **original GPU Max-Tree implementation developed by the authors** of the paper.  
  This code has been included **as-is**, with only minimal updates to dependencies, build scripts, and library versions to ensure compatibility with modern toolchains (CUDA ≥ 12, Conan v2, recent CMake).

- The remaining parts of the repository contain **our contributions**, including:
  - Integration of **CUDA Graphs** into the Max-Tree pipeline
  - Updated build system and benchmarking infrastructure
  - Experimental setup used for the results reported in the paper

All algorithmic details of the original Max-Tree implementation remain unchanged in the `maxtree_original/` directory. Our modifications focus exclusively on execution orchestration, performance optimization, and benchmarking.

## Build Instructions

### Prerequisites

- CMake
- Conan (v2)
- CUDA Toolkit (tested with CUDA ≥ 12)
- Ninja (recommended)

Ensure that nvcc is correctly installed and note its absolute path.

---

### Build Steps

From the root of the repository:


shell
> mkdir build && cd build
> conan install .. --output-folder=. --build=missing -s compiler.cppstd=20
> cmake .. \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12/bin/nvcc \
    -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -G Ninja
> ninja


## Running the Benchmarks

All benchmark executables are generated in the bin/ directory after a successful build.

Before running any benchmark, make sure you have downloaded the test image (see **Test Image** section below) and that you are running from the project root or build/ directory as appropriate.

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


shell
> ./bin/TestMaxtree --no-check ./test.pgm


### 2. Kernel-only benchmark (no memory transfers)

This mode measures **GPU kernel execution time only**, excluding all host–device memory transfers. It is useful for evaluating the intrinsic efficiency of the CUDA kernels without PCIe overhead.

Run:


shell
> ./bin/TestMaxtree --no-check --bench-kernel-only ./test.pgm


## Test Images

The benchmark experiments use grayscale images in **PGM format**.

The primary test image used in the experiments can be downloaded from the following link:

- **Test image (PGM)**:  
  [https://drive.google.com/drive/folders/1zxxVqUco6WYaCPk1M9MeMf9G7E8CPkk7?usp=drive_link]


The original implementation is based on the work:

N. Blin et al., *“Max-Tree Computation on GPUs,”* IEEE Transactions on Parallel and Distributed Systems, 2022.


