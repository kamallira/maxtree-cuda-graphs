#include "tile.cuh"
#include "maxtree.cuh"
#include <cstdint>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cooperative_groups.h>

namespace
{
  struct coord2d_s
  {
    int16_t x;
    int16_t y;
  };


  class LevelRootStack
  {
  public:
    __device__ LevelRootStack(uint8_t* storage)
      : m_roots(storage)
    {
    }

    __device__ void push(int x) { m_roots[m_size++] = x; }
    __device__ int  pop()
    {
      assert(m_size > 0);
      return m_roots[--m_size];
    }
    __device__ int top() const
    {
      assert(m_size > 0);
      return m_roots[m_size - 1];
    }
    __device__ bool empty() const { return m_size == 0; }

  protected:
    uint8_t* m_roots; /* FIXME */
    int m_size = 0;
  };


  // Unstack and attach the parent until the stack get empty or the stack level <= v
  __device__
  int unstack(Tile2DView<info_t> data, LevelRootStack& stack, int w, int col, int v)
  {
    assert(!stack.empty());

    int top_level;
    int r = stack.pop();

    while (!stack.empty() && (top_level = data(col, stack.top()).level) < v)
    {
      auto e = data(col, r);
      e.parent_index = stack.top() * w + col;
      e.parent_level = top_level;
      data(col, r) = e;
      r = stack.pop();
    }

    assert(stack.empty() || data(col, stack.top()).level >= v);
    return r;
  }


  template <int HEIGHT>
  __device__ int compute_mintree_1d(Tile2DView<info_t> buf)
  {
    int   col            = threadIdx.x;
    int   w              = buf.pitch();
    int   height         = buf.height();


    uint8_t storage[HEIGHT];
    LevelRootStack roots(storage);


    roots.push(0);
    for (int i = 1; i < height; i++)
    {
      auto& cur    = buf(col, i);
      auto  top    = buf(col, roots.top());

      int v = cur.level;

      if (top.level > v)
      {
        roots.push(i);
      }
      else if (top.level == v)
      {
        cur.parent_index = roots.top() * w + col;
        cur.parent_level = v;
      }
      else
      {
        int r = unstack(buf, roots, w, col, v);

        if (roots.empty() || buf(col, roots.top()).level > v)
        {
          buf(col, r).parent_index = i * w + col;
          buf(col, r).parent_level = v;
          roots.push(i);
        }
        else
        {
          buf(col, i).parent_index = roots.top() * w + col;
          buf(col, r).parent_index = roots.top() * w + col;
          buf(col, i).parent_level = v;
          buf(col, r).parent_level = v;
        }
      }
    }

    int   root_id    = unstack(buf, roots, w, col, INT32_MAX);
    auto& root       = buf(col, root_id);
    root.parent_index = info_t::kRootParentIndex;
    root.parent_level = info_t::kRootParentLevel;

    return root_id;
  }


} // namespace

// Collaborative loading of data
__device__ __forceinline__
void load_tile_pixel(const uint8_t* __restrict__ input, info_t* __restrict__ tile, int tile_width, int pitch, int x, int y)
{
  info_t p;
  p.level                  = ~(input[y * pitch + x]);
  p.parent_index           = info_t::kRootParentIndex;
  p.parent_level           = info_t::kRootParentLevel;
  assert(info_t::get_level(p.as_uint32()) == p.level);
  assert(info_t::get_par_level(p.as_uint32()) == info_t::kRootParentLevel);
  assert(info_t::get_par_index(p.as_uint32()) == info_t::kRootParentIndex);
  assert(info_t::isRoot(p.as_uint32()));
  tile[y * tile_width + x] = p;
}


__device__ __forceinline__
void load_tile_column(const uint8_t* __restrict__ input, info_t* __restrict__ tile, int tile_width, int pitch, int n)
{
  for (int y = 0; y < n; ++y)
    load_tile_pixel(input, tile, tile_width, pitch, threadIdx.x, y);
}


// Collaborative writing of data
__device__ __forceinline__ void write_tile_pixel(volatile uint32_t* __restrict__ tile, int32_t* __restrict__ parent, int gx,
                                                 int gy, int tile_width, int pitch, int tx, int ty)
{
  auto q      = info_t::from_uint32(tile[ty * tile_width + tx]);
  int  qindex = q.parent_index;
  int  qy     = qindex / tile_width;
  int  qx     = qindex % tile_width;

  parent[(gy + ty) * pitch + (gx + tx)] = (qindex == info_t::kRootParentIndex) ? -1 : ((qy + gy) * pitch + (gx + qx));
  // parent[(gy + ty) * pitch + (gx + tx)] = ty * pitch + tx;
}

__device__ __forceinline__
void write_tile_column(uint32_t* __restrict__ tile, int32_t* __restrict__ parent, int gx, int gy, int tile_width, int pitch, int n)
{
  for (int ty = 0; ty < n; ++ty)
    write_tile_pixel(tile, parent, gx, gy, tile_width, pitch, threadIdx.x, ty);
}

//__device__ unsigned long long timers[4];


struct myclock
{
  __device__ void start()
  {
    if constexpr (ENABLE_KERNEL_TIMERS)
    {
      if (threadIdx.x == 0 && threadIdx.y == 0)
        t0 = clock();
      __syncthreads();
    }
  }

  __device__ void stop(unsigned long long* dst)
  {
    if constexpr (ENABLE_KERNEL_TIMERS)
    {
      if (threadIdx.x == 0 && threadIdx.y == 0)
      {
        unsigned int t1 = clock();
        atomicAdd(dst, t1 - t0);
        t0 = t1;
      }
      __syncthreads();
    }
  }
  unsigned int t0;
};

template <int TILE_WIDTH, int TILE_HEIGHT>
__global__ void compute_maxtree_tile_base_optim(const uint8_t* __restrict__ input_, int32_t* __restrict__ parent, int width,
                                          int height, int pitch, int y_start_index, int connectivity)
{
  assert(connectivity == 4 || connectivity == 8);
  __shared__ uint32_t aux[TILE_HEIGHT][TILE_WIDTH];

  info_t* aux_ = (info_t*)aux;
  int     gx   = blockIdx.x * TILE_WIDTH;
  int     gy   = y_start_index + blockIdx.y * TILE_HEIGHT;
  int     w    = std::min(width - gx, TILE_WIDTH);
  int     h    = std::min(height - gy, TILE_HEIGHT);

  myclock t;
  auto    sAux = Tile2DView<info_t>(aux_, w, h, TILE_WIDTH);

  t.start();

  // 1. Copy to shared memory
  if (threadIdx.x < w && threadIdx.y < h)
    load_tile_pixel(input_ + gy * pitch + gx, aux_, TILE_WIDTH, pitch, threadIdx.x, threadIdx.y);

  __syncthreads();
  t.stop(&timers[0]);

  // 2. Compute the maxtree of the tile
  if (threadIdx.x < w && threadIdx.y < h)
    compute_maxtree_uf_optim(aux_, TILE_WIDTH, connectivity);

  __syncthreads();
  t.stop(&timers[1]);


  // Flatten & Commit
  if (threadIdx.x < w && threadIdx.y < h)
  {
    int qindex = flatten((uint32_t*)aux_, TILE_WIDTH, threadIdx.x, threadIdx.y);

    cooperative_groups::coalesced_threads().sync();

    // 3. Commit into global memory
    int  gidx    = (gy + threadIdx.y) * pitch + (gx + threadIdx.x);
    int  qy      = qindex / TILE_WIDTH;
    int  qx      = qindex % TILE_WIDTH;
    int  qgidx    = (gy + qy) * pitch + (gx + qx);
    parent[gidx] = (qindex == info_t::kRootParentIndex) ? -1 : qgidx;
  }

  __syncthreads();
  t.stop(&timers[4]);
}

template <int TILE_WIDTH, int TILE_HEIGHT>
__global__ void compute_maxtree_tile_base(const uint8_t* __restrict__ input_, int32_t* __restrict__ parent, int width,
                                          int height, int pitch, int y_start_index, int connectivity)
{
  assert(connectivity == 4 || connectivity == 8);
  __shared__ uint32_t aux[TILE_HEIGHT][TILE_WIDTH];

  info_t* aux_ = (info_t*)aux;
  int     gx   = blockIdx.x * TILE_WIDTH;
  int     gy   = y_start_index + blockIdx.y * TILE_HEIGHT;
  int     w    = std::min(width - gx, TILE_WIDTH);
  int     h    = std::min(height - gy, TILE_HEIGHT);

  myclock t;
  auto    sAux = Tile2DView<info_t>(aux_, w, h, TILE_WIDTH);

  t.start();

  // 1. Copy to shared memory
  if (threadIdx.x < w && threadIdx.y < h)
    load_tile_pixel(input_ + gy * pitch + gx, aux_, TILE_WIDTH, pitch, threadIdx.x, threadIdx.y);

  __syncthreads();
  t.stop(&timers[0]);

  // 2. Compute the maxtree of the tile
  if (threadIdx.x < w && threadIdx.y < h)
    compute_maxtree_uf(aux_, TILE_WIDTH, connectivity);

  __syncthreads();
  t.stop(&timers[1]);


  // Flatten & Commit
  if (threadIdx.x < w && threadIdx.y < h)
  {
    int qindex = flatten((uint32_t*)aux_, TILE_WIDTH, threadIdx.x, threadIdx.y);

    cooperative_groups::coalesced_threads().sync();

    // 3. Commit into global memory
    int  gidx    = (gy + threadIdx.y) * pitch + (gx + threadIdx.x);
    int  qy      = qindex / TILE_WIDTH;
    int  qx      = qindex % TILE_WIDTH;
    int  qgidx    = (gy + qy) * pitch + (gx + qx);
    parent[gidx] = (qindex == info_t::kRootParentIndex) ? -1 : qgidx;
  }

  __syncthreads();
  t.stop(&timers[4]);
}

template <int TILE_WIDTH, int TILE_HEIGHT>
__global__ void compute_maxtree_tile_optim_1d_connection(const uint8_t* __restrict__ input_, int32_t* __restrict__ parent_,
                                              int width, int height, int pitch, int y_start_index, int connectivity)
{
    assert(connectivity == 4 || connectivity == 8);
  __shared__ uint32_t aux[TILE_HEIGHT][TILE_WIDTH];

  info_t* aux_ = (info_t*)aux;
  int     gx   = blockIdx.x * TILE_WIDTH;
  int     gy   = y_start_index + blockIdx.y * TILE_HEIGHT;
  int     w    = std::min(width - gx, TILE_WIDTH);
  int     h    = std::min(height - gy, TILE_HEIGHT);

  myclock t;
  auto sAux    = Tile2DView<info_t>(aux_, w, h, TILE_WIDTH);

  t.start();

  // 1. Copy to shared memory
  {
    if (threadIdx.x < w && threadIdx.y < h)
      load_tile_column(input_ + gy * pitch + gx, aux_, TILE_WIDTH, pitch, h);
  }

  __syncthreads();
  t.stop(&timers[0]);


  // 2. Compute the maxtree of the tile
  if (threadIdx.x < w && threadIdx.y == 0)
    compute_mintree_1d<TILE_HEIGHT>(sAux);

  __syncthreads();
  t.stop(&timers[1]);

  // 3. Merge column maxtrees
  if (threadIdx.x < w && threadIdx.y < h)
    merge_columns_with_one_thread_optim(aux_, TILE_WIDTH, h, connectivity);

  __syncthreads();
  t.stop(&timers[2]);


  // Flatten
  if (threadIdx.x < w && threadIdx.y < h)
    flatten_column((uint32_t*)aux_, TILE_WIDTH, h);

  __syncthreads();
  t.stop(&timers[3]);

  // 3. Commit into global memory
  if (threadIdx.x < w && threadIdx.y < h)
    write_tile_column((uint32_t*)aux_, parent_, gx, gy, TILE_WIDTH, pitch, h);

  __syncthreads();
  t.stop(&timers[4]);
}

template <int TILE_WIDTH, int TILE_HEIGHT>
__global__ void compute_maxtree_tile_optim_1d(const uint8_t* __restrict__ input_, int32_t* __restrict__ parent_,
                                              int width, int height, int pitch, int y_start_index, int connectivity)
{
    assert(connectivity == 4 || connectivity == 8);
  __shared__ uint32_t aux[TILE_HEIGHT][TILE_WIDTH];

  info_t* aux_ = (info_t*)aux;
  int     gx   = blockIdx.x * TILE_WIDTH;
  int     gy   = y_start_index + blockIdx.y * TILE_HEIGHT;
  int     w    = std::min(width - gx, TILE_WIDTH);
  int     h    = std::min(height - gy, TILE_HEIGHT);

  myclock t;
  auto sAux    = Tile2DView<info_t>(aux_, w, h, TILE_WIDTH);

  t.start();

  // 1. Copy to shared memory
  {
    if (threadIdx.x < w && threadIdx.y < h)
      load_tile_column(input_ + gy * pitch + gx, aux_, TILE_WIDTH, pitch, h);
  }

  __syncthreads();
  t.stop(&timers[0]);


  // 2. Compute the maxtree of the tile
  if (threadIdx.x < w && threadIdx.y == 0)
    compute_mintree_1d<TILE_HEIGHT>(sAux);

  __syncthreads();
  t.stop(&timers[1]);

  // 3. Merge column maxtrees
  if (threadIdx.x < w && threadIdx.y < h)
    merge_columns_with_one_thread(aux_, TILE_WIDTH, h, connectivity);

  __syncthreads();
  t.stop(&timers[2]);


  // Flatten
  if (threadIdx.x < w && threadIdx.y < h)
    flatten_column((uint32_t*)aux_, TILE_WIDTH, h);

  __syncthreads();
  t.stop(&timers[3]);

  // 3. Commit into global memory
  if (threadIdx.x < w && threadIdx.y < h)
    write_tile_column((uint32_t*)aux_, parent_, gx, gy, TILE_WIDTH, pitch, h);

  __syncthreads();
  t.stop(&timers[4]);
}



