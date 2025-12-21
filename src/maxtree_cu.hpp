#pragma once
#include <mln/bp/tile.hpp>
#include "maxtree_defs.hpp"


///
/// \param chrono If not null, it used to benchmark the GPU kernel time only
void maxtree_cu(const uint8_t* input_buffer, std::ptrdiff_t input_stride, int width, int height, int32_t* parent,
                std::ptrdiff_t parent_stride, uint32_t variant, Chrono* chrono);

void maxtree_cu(const uint16_t* input_buffer, std::ptrdiff_t input_stride, int width, int height, int32_t* parent,
                std::ptrdiff_t parent_stride, uint32_t variant, Chrono* chrono);

void maxtree_allocate_output(int32_t** ptr, int width, int height, int size, int* pitch);
void maxtree_allocate_output(int width, int height, int size, mln::bp::Tile2D<int32_t>& output);



void maxtree_cu(mln::bp::Tile2DView<uint8_t> input, mln::bp::Tile2DView<int32_t> output, uint32_t variation, Chrono* chrono);
void maxtree_cu(mln::bp::Tile2DView<uint16_t> input, mln::bp::Tile2DView<int32_t> output, uint32_t variation, Chrono* chrono);

mln::bp::Tile2D<int32_t>  maxtree_cu(mln::bp::Tile2DView<uint8_t> input, uint32_t variation, Chrono* chrono = nullptr);
mln::bp::Tile2D<int32_t>  maxtree_cu(mln::bp::Tile2DView<uint16_t> input, uint32_t variation, Chrono* chrono = nullptr);


namespace cuda
{
  int  cudaHostRegister(void* ptr, size_t size, unsigned int flags);
  int  cudaHostUnregister(void* ptr);
  void cudaFreeHost(void* ptr);
}
