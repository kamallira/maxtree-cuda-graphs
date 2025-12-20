# Introduction

This folder holds code for running benchmarks adapted from https://gitlab.lre.epita.fr/nblin/max-tree which was used in the paper:

Nicolas Blin, Edwin Carlinet, Florian Lemaitre, Lionel Lacassagne, Thierry Géraud. Max-tree Computation on GPUs. 2022. ⟨hal-03556296⟩

It provides many GPU and CPU implementations for the max-tree construction as well as many GPU variations. This code
should only be used to reproduce the benchmarks, it does not have a stable API.

Configure your environment using the commands below. Make sure the **-DCMAKE_CUDA_COMPILER="** points to the location of your install nvcc

```shell
mkdir build && cd build
conan install .. --output-folder=. --build=missing -s compiler.cppstd=20 
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12/bin/nvcc -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release -G Ninja
ninja
```


In the bin folder. Make sure to download the test image before running the benchmarks

# Running the benchmarks

1. Running benchmarks with memory transfers on:

```shell
>  ./bin/TestMaxtree --no-check  ./ouput.pgm
```

2. Running benchmarks with memory transfers off:

```shell
>  ./bin/TestMaxtree --no-check --bench-kernel-only ./output.pgm
```

# Test Image
The test image can be found here:

https://drive.google.com/file/d/1kb79XAXE1A_ioj8cBvhV2H70QLDZDSXg/view?usp=drive_link




