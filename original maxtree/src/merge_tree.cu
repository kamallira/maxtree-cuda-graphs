#include "tile.cuh"
#include "maxtree.cuh"
#include <cassert>
#include <utility>
#include <cstdio>
#include <optional>
#include <cuda/std/utility>
//#include <cuda/std/optional>
#include <cooperative_groups.h>

namespace
{

  __device__ int uf_find_peak_root(uint32_t* __restrict__ buf, int a, int level_, uint32_t* __restrict__ A)
  {
    assert(a != info_t::kRootParentIndex);


    auto cur   = info_t::from_uint32(buf[a]);
    int  level = level_ < 0 ? cur.level : level_;

    assert(level >= cur.level);

    while (!cur.is_root() && cur.parent_level <= level)
    {
      int old_level  = cur.parent_level;
      a              = cur.parent_index;
      cur            = info_t::from_uint32(buf[a]);

      assert(a != info_t::kRootParentIndex);
      assert(cur.level == old_level);
    }
    if (A)
      *A = cur.as_uint32();
    assert(cur.level <= level);
    return a;
  }

  __device__ int uf_find_level_root(uint32_t* __restrict__ buf, int a, uint32_t* __restrict__ A)
  {
    return uf_find_peak_root(buf, a, -1, A);
  }

  template <class T, class I>
  __device__ __forceinline__
  int uf_find_peak_root_T(const T* __restrict__ input, I* __restrict__ parent, int a, int& level_A, int* parent_A)
  {
    int  level            = level_A;
    int  level_root       = a;
    int  q;

    assert(a >= 0);
    assert(level <= input[a]);

    for (q = parent[a]; q >= 0 && level <= input[q]; q = parent[q])
    {
      assert(input[q] < input[level_root] || (input[level_root] == input[q] && q < level_root));
      level_root = q;
    }

    if (parent_A)
      *parent_A = q;

    level_A = input[level_root];

    assert(level <= level_A);
    assert(level_root >= 0);
    return level_root;
  }


  __device__ __forceinline__
  int uf_find_peak_root(const uint8_t* __restrict__ input, int32_t* __restrict__ parent, int a, int&& level_A, int* parent_A)
  {
    return uf_find_peak_root_T(input, parent, a, level_A, parent_A);
  }

  __device__ __forceinline__
  int uf_find_peak_root(const uint8_t* __restrict__ input, int32_t* __restrict__ parent, int a, int& level_A, int* parent_A)
  {
    return uf_find_peak_root_T(input, parent, a, level_A, parent_A);
  }


  __device__ __forceinline__
  int uf_find_peak_root(const uint16_t* __restrict__ input, int16_t* __restrict__ parent, int a, int&& level_A, int* parent_A)
  {
    return uf_find_peak_root_T(input, parent, a, level_A, parent_A);
  }

  __device__ __forceinline__
  int uf_find_peak_root(const uint16_t* __restrict__ input, int16_t* __restrict__ parent, int a, int& level_A, int* parent_A)
  {
    return uf_find_peak_root_T(input, parent, a, level_A, parent_A);
  }

  __device__ __forceinline__
  int uf_find_peak_root(const uint16_t* __restrict__ input, int32_t* __restrict__ parent, int a, int&& level_A, int* parent_A)
  {
    return uf_find_peak_root_T(input, parent, a, level_A, parent_A);
  }

  __device__ __forceinline__
  int uf_find_peak_root(const uint16_t* __restrict__ input, int32_t* __restrict__ parent, int a, int& level_A, int* parent_A)
  {
    return uf_find_peak_root_T(input, parent, a, level_A, parent_A);
  }

  __device__ __forceinline__
  int uf_find_level_root(const uint8_t* __restrict__ input, int32_t* __restrict__ parent, int a, int* parent_A)
  {
    return uf_find_peak_root(input, parent, a, input[a], parent_A);
  }


  __device__
  void uf_zip(uint32_t* buf, int a, int b)
  {
    using cuda::std::swap;

    auto A = buf[a];
    auto B = buf[b];
    int A_level = info_t::get_level(A);
    int B_level = info_t::get_level(B);

    while (true)
    {
      if (B_level > A_level)
      {
        swap(a, b);
        swap(A_level, B_level);
      }

      a = uf_find_peak_root(buf, a, A_level, &A);
      b = uf_find_peak_root(buf, b, A_level, &B);
      B_level = info_t::get_level(B);

      if (a == b)
        return;

      assert(A_level >= B_level);

      // if A and B are at the same level, ensure a total ordering with
      // localisation (Null pointer are "localized" too)
      if (A_level == B_level && b < a)
        swap(a, b);

      uint32_t newB = info_t::make(B_level, A_level, a);
      uint32_t old = atomicMin_block(buf + b, newB);

      if (info_t::isRoot(old)) /* root */ { return; }

      b       = info_t::get_par_index(old);
      B_level = info_t::get_par_level(old);
    }
    //return
  }

  template <class T, class I>
  __device__
  void uf_zip_global(const T* __restrict__ input, I* __restrict__ parent, int a, int b)
  {
    using cuda::std::swap;
    assert(a >= 0 && b >= 0);



    if (input[b] < input[a])
      swap(a, b);

    int level = input[a];
    int A_parent, B_parent;
    int A_level = level;
    int B_level = level;
    // Find current level component roots and peak component roots
    a = uf_find_peak_root(input, parent, a, A_level, &A_parent);
    b = uf_find_peak_root(input, parent, b, B_level, &B_parent);

    while (a != b)
    {
      assert(A_level <= B_level);

      // Ensure an ordering between pixels based on indexes
      if (A_level == B_level && b < a)
      {
        swap(a,b);
        swap(A_parent, B_parent);
      }
      else
      {
        assert(A_parent == -1 || input[A_parent] < level);
        assert(B_parent == -1 || input[B_parent] < level);
      }

      // connect b to a (b has to be canonical)
      // Merge
      using J = std::make_unsigned_t<I>;
      int old = static_cast<I>(atomicCAS((J*)parent + b, static_cast<J>(B_parent), static_cast<J>(a)));
      if (old == -1) /* root */ { return; }

      int oldv = input[old];
      if (old == B_parent)
      {
        level    = oldv;
        b        = cuda::std::exchange(a, old);

        B_level  = level;
        A_level  = level;
        a = uf_find_peak_root(input, parent, a, A_level, &A_parent);
        b = uf_find_peak_root(input, parent, b, B_level, &B_parent);
      }
      else if (oldv < level) // b is the component root
      {
        B_parent = old;
      }
      else
      {
        B_level = level;
        b = uf_find_peak_root(input, parent, b, B_level, &B_parent);
      }
    }
  }
}


__device__ void merge_columns_with_one_thread_optim(info_t* parent, int pitch, int n, int connectivity)
{
  assert(connectivity == 4 || connectivity == 8);

  if (threadIdx.x == 0)
    return;

 
  for (int y = 0; y < n; ++y)
  {
    int a = y * pitch + threadIdx.x;
    int b = a - 1;

    if (connectivity == 4)
    {
      int m0 = (y > 0) ? std::max(parent[a - pitch].level, parent[b - pitch].level) : 256;
      int m1 = std::max(parent[a].level, parent[b].level);
      //int m2 = ((y+1) < n) ? std::max(parent[a + pitch].level, parent[b + pitch].level) : 256;

      if (m0 <= m1) // || m2 < m1)
        continue;

      uf_zip((uint32_t*)parent, a, b);
    }
    else
    {
      int32_t m = a;
      int32_t M = b;

      if (y > 0)
      {
        int32_t a1 = a - pitch;
        int32_t b1 = b - pitch;
        m = parent[a].level < parent[a1].level ? a : a1;
        M = parent[b].level < parent[b1].level ? b : b1;
      }
   

      if (m == a || M == b)
        uf_zip((uint32_t*)parent, m, M);
    }
  }
}

__device__ void merge_columns_with_one_thread(info_t* parent, int pitch, int n, int connectivity)
{
  assert(connectivity == 4 || connectivity == 8);

  if (threadIdx.x == 0)
    return;

  for (int y = 0; y < n; ++y)
  {
    int a = y * pitch + threadIdx.x;
    int b = a - 1;

    if (connectivity == 8 && y > 0)
      uf_zip((uint32_t*)parent, a, b - pitch);

    uf_zip((uint32_t*)parent, a, b);

    if (connectivity == 8 && y > 0)
      uf_zip((uint32_t*)parent, a - pitch, b);
  }
}

__device__ void compute_maxtree_uf_optim(info_t* parent, int pitch, int connectivity)
{
  assert(connectivity == 4 || connectivity == 8);
  // _   a     ;  _   _  ;    b ↘ _  ;  b 
  // b ↗ _     ;  b → a  ;    _   a  ;  a 

  int a = threadIdx.y * pitch + threadIdx.x;
  int b = a - 1;

  if (threadIdx.y > 0)
  {
    uf_zip((uint32_t*)parent, a, a - pitch);
  }

  if (threadIdx.x == 0)
    return;
  
  if (connectivity == 4)
  {
    int m0 = (threadIdx.y > 0) ? std::max(parent[a - pitch].level, parent[b - pitch].level) : 256;
    int m1 = std::max(parent[a].level, parent[b].level);
    //int m2 = ((threadIdx.y+1) < h) ? std::max(parent[a + pitch].level, parent[b + pitch].level) : 256;

    if (m0 <= m1) // || m2 < m1)
      return;
    uf_zip((uint32_t*)parent, a, b);
  }
  else
  {
    int32_t m = a;
    int32_t M = b;
    if (threadIdx.y > 0)
    {
      int32_t a1 = a - pitch;
      int32_t b1 = b - pitch;
      m = parent[a].level < parent[a1].level ? a : a1;
      M = parent[b].level < parent[b1].level ? b : b1;
    }
    if (m == a || M == b)
      uf_zip((uint32_t*)parent, m, M);
  }
}


__device__ void compute_maxtree_uf(info_t* parent, int pitch, int connectivity)
{
  assert(connectivity == 4 || connectivity == 8);
  // _   a     ;  _   _  ;    b ↘ _  ;  b 
  // b ↗ _     ;  b → a  ;    _   a  ;  a 

  int a = threadIdx.y * pitch + threadIdx.x;
  if (threadIdx.x > 0)
  {
    int b = a - 1;
    // b ↗ a
    if (connectivity == 8 && threadIdx.y > 0)
      uf_zip((uint32_t*)parent, a - pitch, b);

    // b → a
    uf_zip((uint32_t*)parent, a, b);

    // b ↘ a
    if (connectivity == 8 && threadIdx.y > 0)
      uf_zip((uint32_t*)parent, a, b - pitch);
  }

  if (threadIdx.y > 0)
  {
    // b ↑ a
    uf_zip((uint32_t*)parent, a, a - pitch);
  }
}



__device__
int flatten(uint32_t* tile, int pitch, int x, int y)
{
  auto node = info_t::from_uint32(tile[y * pitch + x]);
  if (!node.is_root())
    node.parent_index = uf_find_level_root(tile, node.parent_index, nullptr);

  // Synchronisation is required to avoid Read-after-write hazard inside a warp
  // But actually the data race is benign
  // Same as __syncwarp but only for active threads
  cooperative_groups::coalesced_threads().sync();
  tile[y * pitch + x] = node.as_uint32();
  return node.parent_index;
}

__device__ void flatten_column(uint32_t* tile, int pitch, int n)
{
  for (int y = 0; y < n; ++y)
    flatten(tile, pitch, threadIdx.x, y);
}

template <class T>
__global__
void flatten(const T* __restrict__ input, int32_t* __restrict__ parent, int width, int height, int pitch, int starty)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y + starty;
  int i = y * pitch + x;

  if (x < width && y < height)
  {
    int q = parent[i];
    if (q >= 0)
      q = uf_find_peak_root(input, parent, q, (int)input[q], nullptr);

    // Synchronisation is required to avoid Read-after-write hazard inside a warp
    // But actually the data race is benign
    cooperative_groups::coalesced_threads().sync();
    parent[i] = q;
  }
}

__global__ void merge_maxtree_v_optim(const uint8_t* __restrict__ input, int32_t*__restrict__ parent, int pitch, int width, int height, int tile_size, int connectivity)
{
  int     y = (blockIdx.y * blockDim.y + threadIdx.y + 1) * tile_size;
  int     x = (blockIdx.x * blockDim.x + threadIdx.x);
  int32_t a = (y - 1) * pitch + x;
  int32_t b = (y - 0) * pitch + x;



  if (x >= width || y >= height)
    return;

  if (connectivity == 4)
  {
    int m0 = (x > 0) ? std::min(input[a-1], input[b-1]) : -1;
    int m1 = std::min(input[a], input[b]);
    //int m2 = ((x+1) < width) ? std::min(input[a+1], input[b+1]) : -1;
    
    if (m0 >= m1) // || m2 > m1)
        return;

    uf_zip_global(input, parent, a, b);
  }
  else 
  {
    int32_t m = a;
    int32_t M = b;
    if (x > 0)
    {
      int32_t a1 = a - 1;
      int32_t b1 = b - 1;
      m = input[a] > input[a1] ? a : a1;
      M = input[b] > input[b1] ? b : b1;
    }

    if (m == a || M == b)
      uf_zip_global(input, parent, m, M);
  }


}

// Horizontal merge is called first
__global__ void merge_maxtree_h_optim(const uint8_t* __restrict__ input, int32_t* __restrict__ parent, int pitch, int width, int height, int tile_width, int tile_height, int connectivity)
{
  int     y = (blockIdx.y * blockDim.y + threadIdx.y);
  int     x = (blockIdx.x * blockDim.x + threadIdx.x + 1) * tile_width;
  int32_t a = y * pitch + (x - 1);
  int32_t b = y * pitch + (x - 0);

  if (x >= width || y >= height)
    return;


  if (connectivity == 4)
  {
    if ((y  % tile_height) > 0) // Never connect with an upper tile !
    {
      int m0 = (y > 0) ? std::min(input[a-pitch], input[b-pitch]) : -1;
      int m1 = std::min(input[a], input[b]);
      //int m2 = ((y+1) < height) ? std::min(input[a+pitch], input[b+pitch]) : -1;
  
      if (m0 >= m1) // || m2 > m1)
          return;
    }
    uf_zip_global(input, parent, a, b);
  }
  else
  {
    int32_t m = a;
    int32_t M = b;
    if ((y  % tile_height) > 0) // Never connect with an upper tile !
    {
      int32_t a1 = a - pitch;
      int32_t b1 = b - pitch;
      m = input[a] > input[a1] ? a : a1;
      M = input[b] > input[b1] ? b : b1;
    }

    if (m == a || M == b)
      uf_zip_global(input, parent, m, M);
  }
}

template <class T>
__global__ void merge_maxtree_v(const T* __restrict__ input, int32_t* __restrict__ parent, int pitch, int width, int height, int tile_size, int connectivity)
{
  int     y = (blockIdx.y * blockDim.y + threadIdx.y + 1) * tile_size;
  int     x = (blockIdx.x * blockDim.x + threadIdx.x);
  int32_t a = (y - 1) * pitch + x;
  int32_t b = (y - 0) * pitch + x;

  if (x >= width || y >= height)
    return;

  if (connectivity == 8 && x > 0)
    uf_zip_global(input, parent, a - 1, b);

  uf_zip_global(input, parent, a, b);

  if (connectivity == 8 && x > 0)
    uf_zip_global(input, parent, a, b - 1);
}

template <class T>
__global__ void merge_maxtree_h(const T* __restrict__ input, int32_t* __restrict__ parent, int pitch, int width, int height, int tile_size, int connectivity)
{
  int     y = (blockIdx.y * blockDim.y + threadIdx.y);
  int     x = (blockIdx.x * blockDim.x + threadIdx.x + 1) * tile_size;
  int32_t a = y * pitch + (x - 1);
  int32_t b = y * pitch + (x - 0);

  if (x < width && y < height)
  {
    if (connectivity == 8 && y > 0)
      uf_zip_global(input, parent, a - pitch, b);

    uf_zip_global(input, parent, a, b);

    if (connectivity == 8 && y > 0)
      uf_zip_global(input, parent, a, b - pitch);

  }
}