#pragma once
#include <mln/core/image/ndimage.hpp>


mln::image2d<mln::point2d> make_parent_image(const int* par, std::size_t n, const mln::image2d<int>& node_map);
mln::image2d<mln::point2d> make_parent_image(mln::image2d<mln::point2d> parent, mln::image2d<uint8_t> input);
mln::image2d<mln::point2d> make_parent_image(mln::image2d<mln::ndpoint<2, int16_t>> parent, mln::image2d<uint8_t> input);
mln::image2d<mln::point2d> make_parent_image(mln::image2d<int32_t> parent, mln::image2d<uint8_t> input);
mln::image2d<mln::point2d> make_parent_image(mln::image2d<int32_t> parent, mln::image2d<uint16_t> input);
