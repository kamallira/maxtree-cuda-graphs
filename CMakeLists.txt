cmake_minimum_required(VERSION 3.20)

project(maxtrees-cpp LANGUAGES CXX)

include(FindCUDAToolkit)
enable_language(CUDA)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(fmt REQUIRED)
find_package(pylene REQUIRED)
find_package(GTest REQUIRED)
find_package(benchmark REQUIRED)

add_compile_options(-W -Wall)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --ptxas-options=-v --generate-line-info -lineinfo")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_library(maxtrees
  src/histogram2d.cpp
  src/maxtree2d_salembier.cpp
  src/merge_tree.cpp
  src/maxtree1d.cpp
  src/executor.cpp
  src/executor_maxtree.cpp
  src/tile_pool.cpp
  src/maxtree_cu.cpp
)

add_library(maxtree_cuda src/all.cu)
target_link_libraries(maxtree_cuda PRIVATE fmt::fmt)

target_include_directories(maxtrees INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(maxtrees PRIVATE fmt::fmt pylene::pylene maxtree_cuda)

add_library(TestHelpers OBJECT tests/test_helpers.cpp)
target_link_libraries(TestHelpers PRIVATE pylene::pylene)

add_executable(UTMaxtree tests/test.cpp)
target_link_libraries(UTMaxtree PRIVATE maxtrees pylene::pylene GTest::gtest_main TestHelpers)
add_test(NAME UTMaxtree COMMAND UTMaxtree)

add_executable(TestMaxtree tests/main.cpp)
target_link_libraries(TestMaxtree PRIVATE pylene::pylene maxtrees GTest::gtest benchmark::benchmark TestHelpers)
