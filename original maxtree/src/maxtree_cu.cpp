#include "maxtree_cu.hpp"

void maxtree_cu(mln::bp::Tile2DView<uint8_t> input, mln::bp::Tile2DView<int32_t> out, uint32_t variant, Chrono* chrono)
{
  maxtree_cu(input.data(), input.stride(), input.width(), input.height(), out.data(), out.stride(), variant, chrono);
}

void maxtree_cu(mln::bp::Tile2DView<uint16_t> input, mln::bp::Tile2DView<int32_t> out, uint32_t variant, Chrono* chrono)
{
  maxtree_cu(input.data(), input.stride(), input.width(), input.height(), out.data(), out.stride(), variant, chrono);
}

namespace
{
  template <class T>
  mln::bp::Tile2D<int32_t> maxtree_cu_T(mln::bp::Tile2DView<T> input, uint32_t variant, Chrono* chrono)
  {
    mln::bp::Tile2D<int32_t> out;
    maxtree_allocate_output(input.width(), input.height(), sizeof(T), out);
    maxtree_cu(input.data(), input.stride(), input.width(), input.height(), out.data(), out.stride(), variant, chrono);
    return out;
  }
}

void maxtree_allocate_output(int width, int height, int size, mln::bp::Tile2D<int32_t>& output)
{
  int32_t*       buffer;
  int            pitch;
  ::maxtree_allocate_output(&buffer, width, height, size, &pitch);
  output = mln::bp::Tile2D<int32_t>::acquire(buffer, width, height, pitch * sizeof(int32_t),
                                             cuda::cudaFreeHost);
}

mln::bp::Tile2D<int32_t> maxtree_cu(mln::bp::Tile2DView<uint8_t> input, uint32_t variant, Chrono* chrono)
{
  return maxtree_cu_T(input, variant, chrono);
}

mln::bp::Tile2D<int32_t> maxtree_cu(mln::bp::Tile2DView<uint16_t> input, uint32_t variant, Chrono* chrono)
{
  return maxtree_cu_T(input, variant, chrono);
}


