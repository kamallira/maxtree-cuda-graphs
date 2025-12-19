#include "executor.hpp"
#include "maxtree.hpp"
#include "maxtree_cu.hpp"
#include "test_helpers.hpp"

#include <gtest/gtest.h>
#include <mln/io/imprint.hpp>
#include <mln/core/algorithm/transform.hpp>

#include "tile2d_helpers.hpp"

#include <fmt/ranges.h>

void check_parent_array_sorted(const int* parent, std::size_t n)
{
  for (int i = 0; i < static_cast<int>(n); ++i)
    ASSERT_LE(parent[i], i);
}


TEST(Salembier, simple_c4)
{
  const mln::image2d<uint8_t> input = {{10, 11, 11, 15, 16, 11, +2}, //
                                       {+2, 10, 19, 10, 10, 10, 10}, //
                                       {18, +2, 18, 19, 18, 14, +6}, //
                                       {16, +2, 16, 10, 19, 10, 10}, //
                                       {18, 16, 18, +2, +2, +2, +2}};


  mln::image2d<mln::point2d> ref_parent = {{{6, 2}, {0, 0}, {1, 0}, {1, 0}, {3, 0}, {1, 0}, {6, 0}},
                                           {{6, 0}, {0, 0}, {2, 2}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
                                           {{0, 3}, {6, 0}, {0, 3}, {2, 2}, {2, 2}, {1, 0}, {6, 0}},
                                           {{5, 2}, {6, 0}, {0, 3}, {0, 0}, {2, 2}, {0, 0}, {0, 0}},
                                           {{0, 3}, {0, 3}, {0, 3}, {6, 0}, {6, 0}, {6, 0}, {6, 0}}};


  Executor ex(MaxtreeMethod::Salembier);
  auto     t = execute_sequential(&ex, input, 4);

  // mln::io::imprint(t.nodemap);
  auto parent = make_parent_image(t.parent.get(), t.node_count, t.nodemap);

  // check_parent_array_sorted(par.data(), par.size());
  ASSERT_EQ(parent.as_tile(), ref_parent.as_tile());
}

TEST(Salembier, simple_c8)
{
  const mln::image2d<uint8_t> input = {{10, 11, 11, 15, 16, 11, +2}, //
                                       {+2, 10, 19, 10, 10, 10, 10}, //
                                       {18, +2, 18, 19, 18, 14, +6}, //
                                       {16, +2, 16, 10, 19, 10, 10}, //
                                       {18, 16, 18, +2, +2, +2, +2}};


  mln::image2d<mln::point2d> ref_parent = {{{6, 2}, {0, 0}, {1, 0}, {5, 2}, {3, 0}, {1, 0}, {6, 0}},
                                           {{6, 0}, {0, 0}, {2, 2}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
                                           {{0, 3}, {6, 0}, {0, 3}, {2, 1}, {2, 2}, {1, 0}, {6, 0}},
                                           {{3, 0}, {6, 0}, {0, 3}, {0, 0}, {2, 1}, {0, 0}, {0, 0}},
                                           {{0, 3}, {0, 3}, {0, 3}, {6, 0}, {6, 0}, {6, 0}, {6, 0}}};


  Executor ex(MaxtreeMethod::Salembier);
  auto     t = execute_sequential(&ex, input, 8);

  // mln::io::imprint(t.nodemap);
  auto parent = make_parent_image(t.parent.get(), t.node_count, t.nodemap);

  // check_parent_array_sorted(par.data(), par.size());
  ASSERT_EQ(parent.as_tile(), ref_parent.as_tile());
}


TEST(Maxtree1d, simple_c4)
{

  mln::image2d<uint8_t> input = {{10, 11, 11, 15, 16, 11, +2}, //
                                 {+2, 10, 10, 10, 10, 10, 10}, //
                                 {18, +2, 18, 19, 18, 14, +6}, //
                                 {16, +2, 16, 10, 10, 10, 10}, //
                                 {18, 16, 18, +2, +2, +2, +2}};


  mln::image2d<int> par_ref = {
      {+6, +0, +1, +1, +3, +1, -1}, //
      {-1, +0, +1, +1, +1, +1, +1}, //
      {+1, -1, +5, +2, +2, +6, +1}, //
      {+1, -1, +3, +1, +3, +3, +3}, //
      {+1, +3, +1, -1, +3, +3, +3}, //
  };

  auto parent = mln::image2d<int>(7, 5);


  compute_maxtree_hor_1d(input.as_tile(), parent.as_tile());
  ASSERT_EQ(parent.as_tile(), par_ref.as_tile());
}


//
//


class MaxtreeTest : public testing::Test
{
protected:
  const int                  n = 151;
  mln::image2d<mln::point2d> ref_par;

  enum order_type
  {
    ORDER_MIN,
    ORDER_MAX
  };

public:
  using point2ds = mln::ndpoint<2, int16_t>;

  void check(const Maxtree& t)
  {
    auto parent = make_parent_image(t.parent.get(), t.node_count, t.nodemap);
    // check_parent_array_sorted(par.data(), par.size());

    ASSERT_EQ(parent.as_tile(), ref_par.as_tile());
  }
  
  template <class T>
  void check(mln::bp::Tile2DView<int32_t> par, mln::image2d<T> g)
  {
    int            sizes[2]   = {par.width(), par.height()};
    std::ptrdiff_t strides[2] = {sizeof(int32_t), par.stride()};

    auto impar = mln::image2d<int32_t>::from_buffer(par.data(), sizes, strides);
    // mln::io::imprint(impar);
    auto parent = make_parent_image(impar, g);

    // check_parent_array_sorted(par.data(), par.size());
    ASSERT_EQ(parent.as_tile(), ref_par.as_tile());
  }

};

template <class T>
class LowerMaxtreeTestBase : public MaxtreeTest
{
  public:
    mln::image2d<T> g;

  void SetUp() final
  {
    g.resize(n, n);
    ref_par.resize(n, n);

    for (int y = 0; y < n; y++)
      for (int x = 0; x < n; x++)
      {
        int k           = std::min(x, y);
        g({x, y})       = k;
        ref_par({x, y}) = {k, k};
      }
    for (int x = 1; x < n; x++)
      ref_par({x, x}) = {x - 1, x - 1};
  
	cuda::cudaHostRegister(g.buffer(), g.byte_stride() * g.height(), 0);
  }
    void TearDown()
  {
    cuda::cudaHostUnregister(g.buffer());
  }
};

template <class T>
class UpperMaxtreeTestBase : public MaxtreeTest
{
  public:
    mln::image2d<T> g;

  void SetUp() final
  {
    g.resize(n, n);
    ref_par.resize(n, n);

    for (int y = 0; y < n; y++)
      for (int x = 0; x < n; x++)
      {
        int k           = std::max(x, y);
        g({x, y})       = 255 - k;
        ref_par({x, y}) = {k, 0};
      }
    for (int x = 0; x < n - 1; x++)
      ref_par({x, 0}) = {x + 1, 0};
    ref_par({n - 1, 0}) = {n - 1, 0};
	
	cuda::cudaHostRegister(g.buffer(), g.byte_stride() * g.height(), 0);
  }
    void TearDown()
  {
    cuda::cudaHostUnregister(g.buffer());
  }
};

using LowerMaxtreeTest = LowerMaxtreeTestBase<uint8_t>;
using UpperMaxtreeTest = UpperMaxtreeTestBase<uint8_t>;
using LowerMaxtreeTestHDR = LowerMaxtreeTestBase<uint16_t>;
using UpperMaxtreeTestHDR = UpperMaxtreeTestBase<uint16_t>;



TEST_F(LowerMaxtreeTest, SalembierTiled)
{
  Executor ex(MaxtreeMethod::Salembier);
  auto     t = execute_sequential(&ex, g, 4);
  this->check(t);
}

TEST_F(UpperMaxtreeTest, SalembierTiled)
{
  Executor ex(MaxtreeMethod::Salembier);
  auto     t = execute_sequential(&ex, g, 4);
  this->check(t);
}


TEST_F(LowerMaxtreeTest, Maxtree1DTiled)
{
  Executor ex(MaxtreeMethod::Maxtree1D);
  auto     t = execute_sequential(&ex, g, 4);
  this->check(t);
}

TEST_F(UpperMaxtreeTest, Maxtree1DTiled)
{
  Executor ex(MaxtreeMethod::Maxtree1D);
  auto     t = execute_sequential(&ex, g, 4);
  this->check(t);
}

TEST_F(LowerMaxtreeTest, MaxtreeGPU_Optim_1D)
{
  auto parent = maxtree_cu(g.as_tile(), MAXTREE_OPTIM_1D | MAXTREE_C4);
  this->check(parent, g);
}

TEST_F(UpperMaxtreeTest, MaxtreeGPU_Optim_1D)
{
  auto parent = maxtree_cu(g.as_tile(), MAXTREE_OPTIM_1D | MAXTREE_C4);
  this->check(parent, g);
}


TEST_F(LowerMaxtreeTest, MaxtreeGPU_Base)
{
  auto parent = maxtree_cu(g.as_tile(), MAXTREE_BASE | MAXTREE_C4);
  this->check(parent, g);
}

TEST_F(UpperMaxtreeTest, MaxtreeGPU_Base)
{
  auto parent = maxtree_cu(g.as_tile(), MAXTREE_BASE | MAXTREE_C4);
  this->check(parent, g);
}

TEST_F(LowerMaxtreeTestHDR, MaxtreeGPU_Base)
{
  auto parent = maxtree_cu(g.as_tile(), MAXTREE_BASE | MAXTREE_C4);
  this->check(parent, g);
}

TEST_F(UpperMaxtreeTestHDR, MaxtreeGPU_Base)
{
  auto parent = maxtree_cu(g.as_tile(), MAXTREE_BASE | MAXTREE_C4);
  this->check(parent, g);
}

TEST_F(LowerMaxtreeTestHDR, MaxtreeGPU_Halo)
{
  auto parent = maxtree_cu(g.as_tile(), MAXTREE_BASE | MAXTREE_C4 | GRID_HALO_ON);
  this->check(parent, g);
}

TEST_F(UpperMaxtreeTestHDR, MaxtreeGPU_Halo)
{
  auto parent = maxtree_cu(g.as_tile(), MAXTREE_BASE | MAXTREE_C4 | GRID_HALO_ON);
  this->check(parent, g);
}


TEST_F(MaxtreeTest, CUDA)
{
  auto g = mln::image2d<uint8_t>{
      {10, +2, 18, 16, 18}, //
      {11, 10, +2, +2, 16}, //
      {11, 10, 18, 16, 18}, //
      {15, 10, 19, 10, +2}, //
      {16, 10, 18, 19, +2}, //
      {11, 10, 14, 10, +2}, //
      {+2, 10, +6, 10, +2}, //
  };

  mln::image2d<int> par_ref_v = {
      {6, -1, +1, +1, +1},  //
      {0, +0, -1, -1, +3},  //
      {1, +1, +5, +3, +1},  //
      {1, +1, +2, +1, -1},  //
      {3, +1, +2, +3, +3},  //
      {1, +1, +6, +3, +3},  //
      {-1, +1, +1, +3, +3}, //
  };

  ref_par = mln::image2d<mln::point2d>{
      {{2, 6}, {1, 0}, {3, 0}, {2, 5}, {3, 0}}, //
      {{0, 0}, {0, 0}, {1, 0}, {1, 0}, {3, 0}}, //
      {{0, 1}, {0, 0}, {3, 0}, {3, 0}, {3, 0}}, //
      {{0, 1}, {0, 0}, {2, 2}, {0, 0}, {1, 0}}, //
      {{0, 3}, {0, 0}, {2, 2}, {2, 2}, {1, 0}}, //
      {{0, 1}, {0, 0}, {0, 0}, {0, 0}, {1, 0}}, //
      {{1, 0}, {0, 0}, {1, 0}, {0, 0}, {1, 0}}  //
  };

  {
    auto parent = maxtree_cu(g.as_tile(), MAXTREE_OPTIM_1D | MAXTREE_C4);
    this->check(parent, g);
  }
  {
    auto parent = maxtree_cu(g.as_tile(), MAXTREE_BASE | MAXTREE_C4);
    this->check(parent, g);
  }
  {
    auto parent = maxtree_cu(g.as_tile(), MAXTREE_OPTIM_1D | MAXTREE_C4 | GRID_CONNECTION_OPTIM_ON);
    this->check(parent, g);
  }

  {
    auto parent = maxtree_cu(g.as_tile(), MAXTREE_BASE | MAXTREE_C4 | GRID_CONNECTION_OPTIM_ON);
    this->check(parent, g);
  }
}


TEST_F(MaxtreeTest, CUDA_C8)
{
  auto g = mln::image2d<uint8_t>{
      {10, +2, 18, 16, 18}, //
      {11, 10, +2, +2, 16}, //
      {11, 10, 18, 16, 18}, //
      {15, 10, 19, 10, +2}, //
      {16, 10, 18, 19, +2}, //
      {11, 10, 14, 10, +2}, //
      {+2, 10, +6, 10, +2}, //
  };

  ref_par = mln::image2d<mln::point2d>{
      {{2, 6}, {1, 0}, {3, 0}, {2, 5}, {3, 0}}, //
      {{0, 0}, {0, 0}, {1, 0}, {1, 0}, {3, 0}}, //
      {{0, 1}, {0, 0}, {3, 0}, {3, 0}, {3, 0}}, //
      {{0, 1}, {0, 0}, {2, 2}, {0, 0}, {1, 0}}, //
      {{0, 3}, {0, 0}, {2, 2}, {2, 3}, {1, 0}}, //
      {{0, 1}, {0, 0}, {0, 0}, {0, 0}, {1, 0}}, //
      {{1, 0}, {0, 0}, {1, 0}, {0, 0}, {1, 0}}  //
  };


  {
    auto parent = maxtree_cu(g.as_tile(), MAXTREE_OPTIM_1D | MAXTREE_C8);
    this->check(parent, g);
  }


  {
    auto parent = maxtree_cu(g.as_tile(), MAXTREE_BASE | MAXTREE_C8);
    this->check(parent, g);
  }

  {
    auto parent = maxtree_cu(g.as_tile(), MAXTREE_OPTIM_1D | MAXTREE_C8 | GRID_CONNECTION_OPTIM_ON);
    this->check(parent, g);
  }


  {
    auto parent = maxtree_cu(g.as_tile(), MAXTREE_BASE | MAXTREE_C8 | GRID_CONNECTION_OPTIM_ON);
    this->check(parent, g);
  }
}


TEST_F(MaxtreeTest, CUDA_HDR)
{
  auto g0 = mln::image2d<uint16_t>{
      {10, +2, 18, 16, 18}, //
      {11, 10, +2, +2, 16}, //
      {11, 10, 18, 16, 18}, //
      {15, 10, 19, 10, +2}, //
      {16, 10, 18, 19, +2}, //
      {11, 10, 14, 10, +2}, //
      {+2, 10, +6, 10, +2}, //
  };

  auto g = mln::transform(g0, [](uint16_t x) -> uint16_t { return x * 101; });


  ref_par = mln::image2d<mln::point2d>{
      {{2, 6}, {1, 0}, {3, 0}, {2, 5}, {3, 0}}, //
      {{0, 0}, {0, 0}, {1, 0}, {1, 0}, {3, 0}}, //
      {{0, 1}, {0, 0}, {3, 0}, {3, 0}, {3, 0}}, //
      {{0, 1}, {0, 0}, {2, 2}, {0, 0}, {1, 0}}, //
      {{0, 3}, {0, 0}, {2, 2}, {2, 2}, {1, 0}}, //
      {{0, 1}, {0, 0}, {0, 0}, {0, 0}, {1, 0}}, //
      {{1, 0}, {0, 0}, {1, 0}, {0, 0}, {1, 0}}  //
  };

  cuda::cudaHostRegister(g.buffer(), g.byte_stride() * g.height(), 0);
  {
    auto parent = maxtree_cu(g.as_tile(), MAXTREE_BASE | MAXTREE_C4);
    this->check(parent, g);
  }

  {
    auto parent = maxtree_cu(g.as_tile(), MAXTREE_BASE | MAXTREE_C4 | GRID_HALO_ON);
    this->check(parent, g);
  }

  cuda::cudaHostUnregister(g.buffer());
}