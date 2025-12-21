#pragma once

#include <mln/bp/tile.hpp>
#include <type_traits>
#include <fmt/ostream.h>
#include <iosfwd>

namespace mln::bp
{

  template <class U, class V>
  bool operator==(Tile2DView<U> a, Tile2DView<V> b)
  {
    if (a.width() != b.width())
      return false;
    if (a.height() != b.height())
      return false;

    for (int y = 0; y < a.height(); ++y)
    {
      const U* lineptr_a = a.row(y);
      const V* lineptr_b = b.row(y);
      if (!std::equal(lineptr_a, lineptr_a + a.width(), lineptr_b))
        return false;
    }
    return true;
  }

  template <class U>
void PrintTo(Tile2DView<U> m, std::ostream* os)
{
  *os << "\n";
  for (int y = 0; y < m.height(); ++y) {
    const U* lineptr = m.row(y);
    for (int x = 0; x < m.width() - 1; ++x) {
      if constexpr (std::is_arithmetic_v<U>) {
        fmt::print(*os, FMT_STRING("{:03} "), lineptr[x]);  // padded numbers
      } else {
        fmt::print(*os, "{} ", lineptr[x]);                 // rely on formatter(U)
      }
    }
    if constexpr (std::is_arithmetic_v<U>) {
      fmt::print(*os, FMT_STRING("{:03}"), lineptr[m.width() - 1]);
    } else {
      fmt::print(*os, "{}", lineptr[m.width() - 1]);
    }
    *os << "\n";
  }
}
} // namespace mln::bp


template <class T>
struct fmt::formatter<mln::ndpoint<2, T>> {
  // Consume any format specs until '}'
  constexpr auto parse(format_parse_context& ctx) {
    auto it  = ctx.begin();
    auto end = ctx.end();
    while (it != end && *it != '}') ++it;
    return it;
  }

  template <typename FormatContext>
  auto format(const mln::ndpoint<2, T>& p, FormatContext& ctx) const {
    return fmt::format_to(ctx.out(), " ({:02d},{:02d})", p.x(), p.y());
  }
};


inline bool compare_debug(mln::image2d<uint8_t> input, mln::image2d<mln::point2d> a, mln::image2d<mln::point2d> b)
{
  bool ok = true;
  mln_foreach (auto p, input.domain())
  {
    auto va = a(p);
    auto vb = b(p);

    if (va != vb)
    {
      fmt::print("ERROR (x={},y={},v={})  par_a=({},{},{}) par_b=({}, {}, {})\n", p.x(), p.y(), input(p), va.x(),
                 va.y(), input(va), vb.x(), vb.y(), input(vb));
      ok = false;
    }
  }
  return ok;
}
