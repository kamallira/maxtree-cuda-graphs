# Modernized CUDA Maxtree with Graph Optimization

This repository is a **revitalized and optimized** implementation of the Maxtree algorithm, originally developed by Nicolas Blin(https://gitlab.lre.epita.fr/nblin/max-tree). 

The original codebase had become incompatible with modern build environments due to outdated dependencies and deprecated libraries. This project restores functionality for modern systems and introduces significant performance improvements using **CUDA Graphs**.

### Key Improvements
* **Modernization:** * Updated build system (CMake/Conan) to support current CUDA toolkits.
    * Refactored code to resolve dependency conflicts and deprecated API calls.
* **Optimization:** * Implemented **CUDA Graphs** to replace iterative kernel launches, reducing CPU-side latency.
