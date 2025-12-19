#include "test_helpers.hpp"
#include <mln/core/range/foreach.hpp>
#include <fmt/core.h>

mln::image2d<mln::point2d> make_parent_image(const int* par, std::size_t n, const mln::image2d<int>& node_map)
{
  using mln::point2d;

  std::vector<point2d>                repr(n);
  std::vector<bool>                   has_repr(n, false);
  mln::image2d<point2d> parent;

  mln::resize(parent, node_map);

  mln_foreach(auto px, node_map.pixels())
  {
    auto p = px.point();
    auto id = px.val();
    if (!has_repr[id])
    {
      has_repr[id] = true;
      repr[id] = p;
    }
    parent(p) = repr[id];
  }

  mln_foreach(auto px, node_map.pixels())
  {
    auto p  = px.point();
    auto id = px.val();

    if (parent(p) == p && par[id] != -1)
      parent(p) = repr[par[id]];
  }
  return parent;
}

namespace
{
  /*
  mln::point2d uf_find_levelroot(mln::image2d<mln::point2d> parent, mln::image2d<uint8_t> input, mln::point2d p)
  {
    mln::point2d r = p;
    for (auto q = parent(r); q != r && input(q) == input(r); q = parent(r))
      r = q;
    while (p != r)
      p = std::exchange(parent(p), r);
    return r;
  }
  */

  // Get the levelroot on a compressed parent image
  template <class P, class V>
  mln::point2d uf_get_levelroot(const mln::image2d<P>& parent, const mln::image2d<V>& input, mln::point2d p)
  {
    mln::point2d q = parent(p);
    return (q.x() >= 0 && input(p) == input(q)) ? q : p;
  }


  template <class P>
  mln::point2d uf_get_parent(const mln::image2d<P>& parent, mln::point2d p)
  {
    mln::point2d q = parent(p);
    return (q.x() >= 0) ? q : p;
  }


  template <class V>
  mln::point2d uf_get_levelroot(const mln::image2d<int32_t>& parent, const mln::image2d<V>& input, mln::point2d p)
  {
    int32_t      iq = parent(p);
    if (iq < 0) { return p; }

    mln::point2d q  = parent.point_at_index(iq);
    return (input(p) == input(q)) ? q : p;
  }

  template <>
  mln::point2d uf_get_parent(const mln::image2d<int32_t>& parent, mln::point2d p)
  {
    int32_t      iq  = parent(p);
    return (iq >= 0) ? (mln::point2d)parent.point_at_index(iq) : p;
  }




  template <class P, class V>
  mln::image2d<mln::point2d> make_parent_image_T(mln::image2d<P> parent, mln::image2d<V> input)
  {
    mln::image2d<mln::point2d> newpar;

    constexpr mln::point2d pinf = {INT_MAX, INT_MAX};
    mln::resize(newpar, input).set_init_value(pinf);


    // Pass 1, set in newpar[q] the min of the component where q is the canonical element
    mln_foreach (auto p, input.domain())
    {
      mln::point2d zp = uf_get_levelroot(parent, input, p);
      newpar(zp) = std::min(newpar(zp), p);
    }

    // Pass 2. Set the par of the
    int root = 0;
    mln_foreach (auto p, input.domain())
    {
      mln::point2d zp = uf_get_levelroot(parent, input, p);
      mln::point2d rp = uf_get_levelroot(newpar, input, zp);

      newpar(p) = rp;
      if (p == rp) // In case p, is the new canonical element
      {
        mln::point2d zq = uf_get_parent(parent, zp);
        zq = uf_get_levelroot(newpar, input, zq);
        newpar(p) = zq;
      }
      if (newpar(p) == p)
        root++;
    }
    if (root != 1)
      fmt::print("Detected {} roots\n", root);
    return newpar;
  }

} // namespace



mln::image2d<mln::point2d> make_parent_image(mln::image2d<mln::point2d> parent, mln::image2d<uint8_t> input)
{
  return make_parent_image_T(parent, input);
}

mln::image2d<mln::point2d> make_parent_image(mln::image2d<mln::ndpoint<2, int16_t>> parent, mln::image2d<uint8_t> input)
{
  return make_parent_image_T(parent, input);
}

mln::image2d<mln::point2d> make_parent_image(mln::image2d<int32_t> parent, mln::image2d<uint8_t> input)
{
  return make_parent_image_T(parent, input);
}

mln::image2d<mln::point2d> make_parent_image(mln::image2d<int32_t> parent, mln::image2d<uint16_t> input)
{
  return make_parent_image_T(parent, input);
}

