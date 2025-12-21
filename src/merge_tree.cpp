#include <mln/bp/tile.hpp>
#include <mln/bp/functional.hpp>

#include <vector>
#include <algorithm>
#include <atomic>

namespace
{
  using node_id_t = int;

  // Find the root of the level component of \p a
  template <class V>
  int uf_find_level_root(int* __restrict parent, const V* __restrict levels, int a, bool path_compress = true)
  {
    assert(a >= 0);

    int  level_root = a;
    auto cur_level  = levels[a];


    for (int q = parent[a]; q >= 0 && levels[q] == cur_level; q = parent[q])
      level_root = q;

    // Path compression
    if (path_compress)
    {
      while (a != level_root)
        a = std::exchange(parent[a], level_root);
    }
    assert(level_root >= 0);
    return level_root;
  }

  template <class V>
  [[gnu::noinline]]
  int uf_find_level_root_concurrent(int* __restrict parent, const V* __restrict levels, int a, bool path_compress, int start, int end)
  {
    assert(a >= 0);

    int  level_root;
    auto cur_level  = levels[a];

    {
      int q = a;
      do {
        level_root = q;
        q = std::atomic_ref<int>(parent[q]).load(std::memory_order_relaxed);
      } while (q >= 0 && levels[q] == cur_level);
    }

    // Path compression
    if (path_compress && level_root != a)
    {
      while (true)
      {
        std::atomic_ref<int> p{parent[a]};
        int q = p.load(std::memory_order_relaxed);
        if (q == level_root)
          break;

        if (start <= a && a < end)
          p.store(level_root, std::memory_order_relaxed);
        a = q;
      };
    }

    assert(level_root >= 0);
    return level_root;
  }



  // Find the root of the peak component of \p a at level \p lambda
  template <class V>
  int uf_find_peak_root(int* __restrict parent, const V* __restrict levels, int a, int lambda,  bool path_compress = true)
  {
    assert(a >= 0);
    assert(lambda <= levels[a]);
    for (int q = parent[a]; q >= 0 && lambda <= levels[q]; q = parent[q])
    {
      if (path_compress)
        q = uf_find_level_root(parent, levels, q, true);
      a = q;
    }
    assert(lambda <= levels[a]);
    return a;
  }

  // Merge (zip) two trees by linking the pixels \p a and \p b
  template <class V>
  void uf_zip(int* parent, V* levels, int a, int b)
  {
    auto cmp_b_a = levels[b] <=> levels[a];
    if (cmp_b_a < 0)
      std::swap(a,b);

    a = uf_find_level_root(parent, levels, a);
    b = uf_find_level_root(parent, levels, b);

    if (cmp_b_a != 0)
      b = uf_find_peak_root(parent, levels, b, levels[a]);

    while (a != b)
    {
      assert(levels[a] <= levels[b]);

      b = std::exchange(parent[b], a); // Union
      if (b < 0) /* root case */ { return; }
      std::swap(a, b);
      a = uf_find_level_root(parent, levels, a);
      b = uf_find_peak_root(parent, levels, b, levels[a]);
    }
  }



  // template <class V>
  // void uf_zip(int* parent, V* levels, int a, int b)
  // {
  //   if (levels[b] < levels[a])
  //     std::swap(a,b);

  //   a = uf_find_level_root(parent, levels, a);
  //   b = uf_find_level_root(parent, levels, b);

  //   while (a != b)
  //   {
  //     assert(levels[a] <= levels[b]);

  //     int q = parent[b];
  //     if (q < 0) /* root case */ { parent[b] = a; return; }

  //     q = uf_find_level_root(parent, levels, q);
  //     if (levels[a] <= levels[q]) { b = q; }
  //     else { parent[b] = a; b = a; a = q; }
  //   }
  // }


  template <class T>
  inline void uf_zip_halo(int* parent, T* levels, int a, int b)
  {
    assert(a >= 0 && b >= 0);
    assert(levels[a] == levels[b]);
    uf_zip(parent, levels, a, b);
  }

}


void merge_halo(mln::bp::Tile2DView<int> A_nodemap, //
                mln::bp::Tile2DView<int> B_nodemap, //
                int*                     parent,    //
                uint8_t*                 levels,    //
                int                      axis)
{
  if (axis == 1)
  {
    assert(A_nodemap.height() == B_nodemap.height());

    int w = A_nodemap.width();
    for (int y = 0; y < A_nodemap.height(); ++y)
      uf_zip_halo(parent, levels, A_nodemap(w-1, y), B_nodemap(0, y));
  }
  else
  {
    assert(A_nodemap.width() == B_nodemap.width());

    int h = A_nodemap.height();
    for (int x = 0; x < A_nodemap.width(); ++x)
      uf_zip_halo(parent, levels, A_nodemap(x, h-1), B_nodemap(x, 0));
  }
}

void merge_halo(int*        A_nodemap, //
                int*        B_nodemap, //
                std::size_t n,         //
                int*        parent,    //
                uint8_t*    levels)
{
  for (std::size_t i = 0; i < n; ++i)
    uf_zip_halo(parent, levels, A_nodemap[i], B_nodemap[i]);
}


void merge_halo(int         gid_1,  //
                int         gid_2,  //
                std::size_t n,      //
                int*        parent, //
                uint8_t* levels, int step)
{
  for (std::size_t i = 0; i < n; ++i)
    uf_zip_halo(parent, levels, gid_1 + i * step, gid_2 + i * step);
}


void canonicalize_parent(int* __restrict parent, uint8_t* __restrict levels, std::size_t begin, std::size_t end)
{
  static_assert(std::atomic_ref<int>::required_alignment == sizeof(int));
  // Ensure parent[x] is canonical
  for (std::size_t i = begin; i < end; ++i)
  {
    auto p = std::atomic_ref<int>{parent[i]};
    int  q = p.load(std::memory_order_relaxed);
    if (q == INT32_MIN)
      break;
    if (q >= 0)
      p.store(uf_find_level_root_concurrent(parent, levels, q, false, 0, INT32_MAX), std::memory_order_relaxed);
  }
}

void canonicalize_parent(int* parent, uint8_t* levels, int width, int height, int global_id, int pitch_index)
{
  for (int y = 0; y < height; ++y)
  {
    int base_id = global_id + y * pitch_index;
    canonicalize_parent(parent, levels, base_id, base_id + width);
  }
}


void canonicalize_nodemap(mln::bp::Tile2DView<int> nodemap, int* __restrict parent, uint8_t* __restrict levels)
{
  // Ensure nodemap[x] is a real node
  for (int y = 0; y < nodemap.height(); ++y)
  {
    auto r = nodemap.row(y);
    for (int x = 0; x < nodemap.width(); ++x)
    {
      int rx = r[x];
      int qx = parent[rx];
      if (qx >= 0 && levels[rx] == levels[qx])
        r[x] = qx;
    }
  }
}

void canonicalize_nodemap_autoindex(mln::bp::Tile2DView<int> nodemap, int* __restrict parent, uint8_t* __restrict levels, int global_index, int index_pitch)
{
  // Ensure nodemap[x] is a real node
  for (int y = 0; y < nodemap.height(); ++y)
  {
    int* __restrict r = nodemap.row(y);
    for (int x = 0; x < nodemap.width(); ++x)
    {
      int rx = global_index + y * index_pitch + x;
      int qx = parent[rx];
      r[x]   = (qx >= 0 && levels[rx] == levels[qx]) ? qx : rx;
    }
  }
}


void merge_rows(int* __restrict parent, uint8_t* __restrict levels,  int width, int height, int global_id, int index_pitch, int connectivity)
{
  for (int y = 1; y < height; ++y)
  {
    int prev_index = global_id + (y - 1) * index_pitch;
    int base_index = global_id + y * index_pitch;
    for (int x = 0; x < width; ++x)
    {
      uf_zip(parent, levels, prev_index + x, base_index + x);

      // HALO Tricks no need for higher connectivity
  
      //if (connectivity > 4 && x >= 1)
      //  uf_zip(parent, levels, prev_index + x - 1, base_index + x);
      //if (connectivity > 4 && x < (width - 1))
      //  uf_zip(parent, levels, prev_index + x + 1, base_index + x);
    }
  }
}
