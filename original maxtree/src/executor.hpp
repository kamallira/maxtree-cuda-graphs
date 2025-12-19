#pragma once

#include <memory>
#include <span>

#include <mln/core/box.hpp>
#include <mln/bp/tile.hpp>
#include <mln/core/image/ndimage.hpp>
#include "tile_pool.hpp"

enum class Algorithm {
  Maxtree_Salembier
};

struct Maxtree
{
  mln::image2d<int>           nodemap;
  std::unique_ptr<int[]>      parent;
  std::unique_ptr<uint8_t[]>  levels;

  std::size_t node_count;
};

struct LocalMaxtree
{
  mln::bp::Tile2DView<int>     nodemap; // Local View in the global nodemap
  int*                         parent;  // Pointer to the global parent array
  uint8_t*                     levels;  // Pointer to the global levels array
  std::size_t                  first;   // Position of the first node in the array
  std::size_t                  count;   // Number of nodes
};

struct MaxtreeData
{
  mln::bp::Tile2DView<uint8_t> input;   // Input image
  mln::bp::Tile2DView<int>     nodemap; // Local View in the global nodemap
  int*                         parent;  // Pointer to the global parent array
  uint8_t*                     levels;  // Pointer to the global levels array
  mln::bp::Tile2D<int>         halo_cols; // Nodemap for halo columns (nx X H)
  mln::bp::Tile2D<int>         halo_rows; // Nodemap for halo columns (ny X W)

  // Number of tiles
  int nx;
  int ny;
  int halo_width;   // Full image width with halo columns repeated twice
  int halo_height;  // Full image height with halo rows repeated twice
};


enum class MaxtreeMethod
{
  Salembier,
  Maxtree1D,
};


// Base class for parallel filter
class Executor
{
public:
  static constexpr int MAX_TILE_WIDTH = 256;
  static constexpr int MAX_TILE_HEIGHT = 256;

  struct Impl
  {
    Impl(MaxtreeData* mt, TileMemoryPoolT<int>* pool, MaxtreeMethod method);

    // Copy the executor
    Impl(const Impl&) = default;

    // Compute the maxtree on the input tile
    // The nodemap, parent, ... can be stored in the local memory (or not), one has to call
    //  CommitToGlocalMemory to copy the local data to global memory afterward
    void ComputeLocalMaxtree(int bx, int by, int connectivity) { std::invoke(m_ComputeLocalMaxtree_ptr, this, bx, by, connectivity); }

    // Merge the local tree to the `dst` Maxtree (which can already be in global memory)
    // The local nodemap is modified
    void MergeMaxtrees(int bx, int by, int nbloc, int axis)  { std::invoke(m_merge_maxtrees_ptr, this, bx, by, nbloc, axis); }


    // Canocalize parent & nodemap
    void Canonicalize(int bx, int by) { std::invoke(m_canocalize_ptr, this, bx, by); }

  private:
    void (Impl::*m_ComputeLocalMaxtree_ptr)(int, int, int);
    void (Impl::*m_canocalize_ptr)(int, int);
    void (Impl::*m_merge_maxtrees_ptr)(int, int, int, int);
    
    void ComputeLocalMaxtree_Salembier(int bx, int by, int connectivity);
    void ComputeLocalMaxtree_1D(int bx, int by, int connectivity);

    void CommitBorderToGlobalMemory(mln::bp::Tile2DView<int> aux, int bx, int by);
    void CommitTileToGlobalMemory(mln::bp::Tile2DView<int> aux, int bx, int by);

    void canonicalize_parent_and_nodemap(int bx, int by);
    void canonicalize_parent_and_nodemap_auto(int bx, int by);

    void MergeMaxtrees_NoNodemap(int bx, int by, int nbloc, int axis);
    void MergeMaxtrees_Nodemap(int bx, int by, int nbloc, int axis);

  protected:
    MaxtreeData*          m_data;
    TileMemoryPoolT<int>* m_pool;
  };

  Impl create(Maxtree& t, const mln::ndbuffer_image& input);

  Executor(MaxtreeMethod method);

  const MaxtreeData* data() const { return &m_data; }

private:
  MaxtreeData               m_data;
  TileMemoryPoolT<int>      m_pool;
  MaxtreeMethod             m_method;
};



Maxtree execute_parallel(Executor* executor, const mln::ndbuffer_image& input, int connectivity);
Maxtree execute_sequential(Executor* executor, const mln::ndbuffer_image& input, int connectivity);
