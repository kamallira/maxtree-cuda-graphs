#include "executor.hpp"
#include <mln/core/image/ndimage.hpp>
#include "executor_maxtree.hpp"
#include "maxtree.hpp"



#include <thread>


#include <tbb/blocked_range2d.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_for.h>
#include <tbb/global_control.h>
#include <fmt/core.h>

class ExecutorWrapper
{
public:
  ExecutorWrapper(Executor::Impl* exec, int connectivity)
    : m_impl{exec}
    , m_connectivity{connectivity}
  {
  }

  ExecutorWrapper(ExecutorWrapper& other, tbb::split)
    : m_impl{other.m_impl}
    , m_connectivity{other.m_connectivity}
  {
  }


  void operator()(tbb::blocked_range2d<int> item)
  {
    assert(item.rows().size() == 1);
    assert(item.cols().size() == 1);

    by = item.rows().begin();
    bx = item.cols().begin();
    x_count = 1;
    y_count = 1;
    //fmt::print("Apply bloc ({},{})\n", bx, by);
    m_impl->ComputeLocalMaxtree(bx, by, m_connectivity);
  }

  void join(ExecutorWrapper& other)
    {
      assert(bx == other.bx || by == other.by);

      //fmt::print("Merge bloc {}x{}@({},{}) with {}x{}@({},{})\n", x_count, y_count, bx, by, 
      //           other.x_count, other.y_count, other.bx, other.by);
      if (by == other.by)
      {
        assert(y_count == other.y_count);
        m_impl->MergeMaxtrees(bx, by - y_count + 1, y_count, 0);
        x_count += other.x_count;
      }
      else
      {
        assert(x_count == other.x_count);
        m_impl->MergeMaxtrees(bx - x_count + 1, by, x_count, 1);
        y_count += other.y_count;
      }
      bx = other.bx;
      by = other.by;
    }

private:
  Executor::Impl* m_impl;
  int m_connectivity;


  int by;
  int bx;
  int x_count;
  int y_count;
};


Executor::Impl Executor::create(Maxtree& t, const mln::ndbuffer_image& input_)
{
  auto input = *input_.cast_to<uint8_t, 2>();

  int w = input.width();
  int h = input.height();
  int nx = 1, ny = 1;
  if (w > MAX_TILE_WIDTH)
    nx = (w + MAX_TILE_WIDTH - 2) / (MAX_TILE_WIDTH - 1);
  if (h > MAX_TILE_HEIGHT)
    ny = (h + MAX_TILE_HEIGHT - 2) / (MAX_TILE_HEIGHT - 1);

  int kTileSize = MAX_TILE_WIDTH * MAX_TILE_HEIGHT;


  t.nodemap.resize(input.domain());
  t.parent.reset(new int[kTileSize * nx * ny]);
  t.levels.reset(new uint8_t[kTileSize * nx * ny]);
  t.node_count = kTileSize * nx * ny;

  this->m_data.input   = input.as_tile();
  this->m_data.nodemap = t.nodemap.as_tile();
  this->m_data.parent  = t.parent.get();
  this->m_data.levels  = t.levels.get();
  this->m_data.nx = nx;
  this->m_data.ny = ny;
  this->m_data.halo_width = w + nx - 1;
  this->m_data.halo_height = h + ny - 1;

  if (nx > 1)
    this->m_data.halo_cols = mln::bp::Tile2D<int>(m_data.halo_height, 2 * nx);
  if (ny > 1)
    this->m_data.halo_rows = mln::bp::Tile2D<int>(m_data.halo_width, 2 * ny);

  return Executor::Impl(&this->m_data, &this->m_pool, this->m_method);
}


namespace
{
  Maxtree execute(Executor* executor, const mln::ndbuffer_image& input, int connectivity)
  {
    Maxtree t;

    auto exec = executor->create(t, input);

    auto                      mt_data = executor->data();
    tbb::blocked_range2d<int> grid{0, mt_data->ny, 0, mt_data->nx};


    ExecutorWrapper body(&exec, connectivity);
   tbb::parallel_reduce(grid, body, tbb::simple_partitioner{});


    // Canonicalize parent array & node map
    tbb::parallel_for(grid, [e = &exec](tbb::blocked_range2d<int> item) {
      auto by = item.rows().begin();
      auto bx = item.cols().begin();
      e->Canonicalize(bx, by);
    }, tbb::simple_partitioner{});

    return t;
  }
}

Maxtree execute_parallel(Executor* executor, const mln::ndbuffer_image& input, int connectivity)
{
  return ::execute(executor, input, connectivity);
}

Maxtree execute_sequential(Executor* executor, const mln::ndbuffer_image& input, int connectivity)
{
  tbb::global_control control(
    tbb::global_control::max_allowed_parallelism,
    1);
  return ::execute(executor, input, connectivity);
}



Executor::Executor(MaxtreeMethod method)
  : m_pool(MAX_TILE_WIDTH + 2, MAX_TILE_HEIGHT + 2),
    m_method(method)
{
}
