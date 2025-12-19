#pragma once

#include <mln/bp/tile.hpp>
#include <vector>

namespace mln::data
{

  template <class T>
  std::vector<std::size_t> histogram(mln::bp::Tile2DView<T> in);



  extern template
  std::vector<std::size_t> histogram(mln::bp::Tile2DView<uint8_t> in);
}
