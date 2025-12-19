# Introduction

This folder holds code for running benchmarks adapted from https://gitlab.lre.epita.fr/nblin/max-tree which was used in the paper:

Nicolas Blin, Edwin Carlinet, Florian Lemaitre, Lionel Lacassagne, Thierry Géraud. Max-tree Computation on GPUs. 2022. ⟨hal-03556296⟩

It provides many GPU and CPU implementations for the max-tree construction as well as many GPU variations. This code
should only be used to reproduce the benchmarks, it does not have a stable API (we are currently working on a more
stable API with only GPU's code).


```shell
mkdir build && cd build
conan install .. --output-folder=. --build=missing -s compiler.cppstd=20 
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12/bin/nvcc -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release -G Ninja
ninja
```


In the bin folder

# Running the benchmarks

1. Running benchmarks with memory transfers on:

```shell
>  ./bin/TestMaxtree --no-check  ../ouput.pgm
```

2. Running benchmarks with memory transfers off:

```shell
>  ./bin/TestMaxtree --no-check --bench-kernel-only ./BHdV_PL_ATL20Ardt_1926_0004.pgm
```


