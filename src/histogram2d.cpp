#include <cassert>
#include "histogram2d.hpp"

#include <mln/bp/tile.hpp>
#include <vector>
#include <algorithm>

namespace mln::data
{
  template <class T>
  std::vector<std::size_t> histogram(mln::bp::Tile2DView<T> in)
  {
    std::vector<std::size_t> hist(256,0);

    for (int y = 0; y < in.height(); ++y)
    {
      auto line = in.row(y);
      std::for_each(line, line + in.width(), [h = hist.data()](uint8_t x) { h[x]++; });
    }

    return hist;
  }

  template
  std::vector<std::size_t> histogram(mln::bp::Tile2DView<uint8_t> in);

}
