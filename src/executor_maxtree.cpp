#include "executor.hpp"

#include "maxtree.hpp"
#include <mln/bp/fill.hpp>
#include <mln/bp/copy.hpp>
#include <mln/bp/functional.hpp>
#include <mln/core/extension/padding.hpp>


#include <fmt/core.h>

Executor::Impl::Impl(MaxtreeData* mt, TileMemoryPoolT<int>* pool, MaxtreeMethod method)
  //  : m_aux(Executor::MAX_TILE_WIDTH + 2, Executor::MAX_TILE_HEIGHT + 2)
  : m_data{mt}
  , m_pool{pool}
{
  switch (method)
  {
  case MaxtreeMethod::Salembier:
    m_ComputeLocalMaxtree_ptr = &Impl::ComputeLocalMaxtree_Salembier;
    m_canocalize_ptr = &Impl::canonicalize_parent_and_nodemap;
    m_merge_maxtrees_ptr = &Impl::MergeMaxtrees_Nodemap;
    break;
  case MaxtreeMethod::Maxtree1D:
    m_ComputeLocalMaxtree_ptr = &Impl::ComputeLocalMaxtree_1D;
    m_canocalize_ptr = &Impl::canonicalize_parent_and_nodemap_auto;
    m_merge_maxtrees_ptr = &Impl::MergeMaxtrees_NoNodemap;
    break;
  }
}


void Executor::Impl::ComputeLocalMaxtree_Salembier(int bx, int by, int connectivity)
{
  int TILE_SIZE = Executor::MAX_TILE_WIDTH * Executor::MAX_TILE_HEIGHT;
  int w = m_data->nodemap.width();
  int h = m_data->nodemap.height();
  int parent_offset = by * m_data->nx * TILE_SIZE  + bx * TILE_SIZE;

  // Note: beware the halo when locating (x,y)
  int x = bx * (Executor::MAX_TILE_WIDTH - 1);
  int y = by * (Executor::MAX_TILE_HEIGHT - 1);
  int width = std::min(Executor::MAX_TILE_WIDTH, w - x);
  int height = std::min(Executor::MAX_TILE_HEIGHT, h - y);

  mln::bp::Tile2D<int> aux{};
  m_pool->acquire(aux);

  // Compute the local maxtree
  {
    {
      auto nodemap = aux.clip(0, 0, width+2, height+2);
      int  borders[2][2] = {{1, 1}, {1, 1}};
      mln::bp::fill(nodemap, -1);
      mln::pad(nodemap, mln::PAD_CONSTANT, borders, INT32_MAX);
    }

    auto input   = m_data->input.clip(x, y, width, height);
    auto nodemap = aux.clip(1, 1, width, height);
    int  new_offset = maxtree2d_salembier(input, nodemap, m_data->parent, m_data->levels, connectivity, parent_offset);
    int  count      = new_offset - parent_offset;

    // Set the sentinel value
    if (count < TILE_SIZE)
      m_data->parent[parent_offset + count] = INT32_MIN;
  }

  // Copy local nodemap to global memory
  {
    this->CommitTileToGlobalMemory(aux, bx, by);
    this->CommitBorderToGlobalMemory(aux, bx, by);
  }


  // Release extra mem for processing
  m_pool->release(aux);
}


void Executor::Impl::ComputeLocalMaxtree_1D(int bx, int by, int connectivity)
{
  int kTileSize = Executor::MAX_TILE_WIDTH * Executor::MAX_TILE_HEIGHT;
  int w = m_data->nodemap.width();
  int h = m_data->nodemap.height();


  // Note: beware the halo when locating (x,y)
  int x = bx * (Executor::MAX_TILE_WIDTH - 1);
  int y = by * (Executor::MAX_TILE_HEIGHT - 1);
  int width = std::min(Executor::MAX_TILE_WIDTH, w - x);
  int height = std::min(Executor::MAX_TILE_HEIGHT, h - y);

  int global_id = by * m_data->nx * kTileSize  + bx * kTileSize;
  {
    auto                         input = m_data->input.clip(x, y, width, height);
    mln::bp::Tile2DView<int>     parent(m_data->parent + global_id, width, height, Executor::MAX_TILE_WIDTH * sizeof(int));
    mln::bp::Tile2DView<uint8_t> levels(m_data->levels + global_id, width, height, Executor::MAX_TILE_WIDTH * sizeof(uint8_t));

    // Compute the local maxtree horizontally
    compute_maxtree_hor_1d(input, parent, global_id, Executor::MAX_TILE_WIDTH);

    // Copy the tile values
    mln::bp::copy(input, levels);
  }

  // Merge the lines
  merge_rows(m_data->parent, m_data->levels, width, height, global_id, Executor::MAX_TILE_WIDTH, connectivity);

}



void Executor::Impl::MergeMaxtrees_Nodemap(int bx, int by, int nbloc, int axis)
{
  int y = by * Executor::MAX_TILE_HEIGHT;
  int x = bx * Executor::MAX_TILE_WIDTH;

  int* A, *B;
  int n;
  if (axis == 0) // Merge region on the right border
  {
    A = m_data->halo_cols.row(2 * bx) + y;
    B = m_data->halo_cols.row(2 * bx + 1) + y;
    n = std::min(nbloc * Executor::MAX_TILE_HEIGHT, m_data->halo_height - y);
  }
  else // Merge region on the bottom border
  {
    A = m_data->halo_rows.row(2 * by) + x;
    B = m_data->halo_rows.row(2 * by + 1) + x;
    n = std::min(nbloc * Executor::MAX_TILE_WIDTH, m_data->halo_width - x);
  }
  merge_halo(A, B, n, m_data->parent, m_data->levels);
}

void Executor::Impl::MergeMaxtrees_NoNodemap(int bx, int by, int nbloc, int axis)
{
  int y0 = by * Executor::MAX_TILE_HEIGHT;
  int x0 = bx * Executor::MAX_TILE_WIDTH;
  int kTileSize = Executor::MAX_TILE_HEIGHT * Executor::MAX_TILE_HEIGHT;
  int kTileRowSize = kTileSize * m_data->nx;

  auto get_global_index = [kTileSize, kTileRowSize](int bloc_x, int bloc_y, int x, int y) {
    return bloc_y * kTileRowSize + bloc_x * kTileSize + y * Executor::MAX_TILE_WIDTH + x;
  };


  if (axis == 0) // Merge region on the right border
  {
    for (int y = 0; y < nbloc; ++y, y0 += Executor::MAX_TILE_HEIGHT)
    {
      int gid_1 = get_global_index(bx + 0, by + y, Executor::MAX_TILE_WIDTH - 1, 0);
      int gid_2 = get_global_index(bx + 1, by + y, 0, 0);
      int n     = std::min(Executor::MAX_TILE_HEIGHT, m_data->halo_height - y0);
      merge_halo(gid_1, gid_2, n, m_data->parent, m_data->levels, Executor::MAX_TILE_WIDTH);
    }
  }
  else // Merge region on the bottom border
  {
    for (int x = 0; x < nbloc; ++x, x0 += Executor::MAX_TILE_WIDTH)
    {
      int gid_1 = get_global_index(bx + x, by + 0, 0, Executor::MAX_TILE_HEIGHT - 1);
      int gid_2 = get_global_index(bx + x, by + 1, 0, 0);
      int n     = std::min(Executor::MAX_TILE_WIDTH, m_data->halo_width - x0);
      merge_halo(gid_1, gid_2, n, m_data->parent, m_data->levels, 1);
    }
  }

}



  /*

void Executor::Impl::MergeLocalMaxtree(mln::bp::Tile2DView<int> dst, int axis)
{
  //auto* other = static_cast<Executor::Impl*>(other_);
  auto nodemap = m_aux.clip(1, 1, m_out->nodemap.width(), m_out->nodemap.height());
  merge_halo(dst, nodemap,                 //
             m_out->parent, m_out->levels, //
             axis);
}

void Executor::Impl::MergeLocalMaxtree(std::span<int> dst, int axis)
{
  int w = m_out->nodemap.height();
  int h = m_out->nodemap.width();
  auto nodemap = m_aux.clip(1, 1, w, h);


  std::unique_ptr<int[]> aux_data_container;
  int*                   aux_data;
  int                    n = axis == 0 ? h : w;
  if (axis == 0)
  {
    aux_data_container.reset(new int[h]);
    aux_data = aux_data_container.get();
    for (int y = 0; y < h; ++y)
      aux_data[y] = nodemap(0, y);
  }
  else
  {
    aux_data = nodemap.row(0);
  }

  assert(dst.size() == n);
}

  */


void Executor::Impl::CommitTileToGlobalMemory(mln::bp::Tile2DView<int> aux, int bx, int by)
{
  int w = m_data->nodemap.width();
  int h = m_data->nodemap.height();

  // Note: beware the halo when locating (x,y)
  int x      = bx * (Executor::MAX_TILE_WIDTH - 1);
  int y      = by * (Executor::MAX_TILE_HEIGHT - 1);
  int width  = (bx != (m_data->nx - 1)) ? (Executor::MAX_TILE_WIDTH - 1) : (w - x);  // Skip the right halo
  int height = (by != (m_data->ny - 1)) ? (Executor::MAX_TILE_HEIGHT - 1) : (h - y); // Skip the bottom halo


  auto local_nodemap = aux.clip(1, 1, width, height);
  auto global_nodemap = m_data->nodemap.clip(x, y, width, height);
  mln::bp::copy(local_nodemap, global_nodemap);
}


void Executor::Impl::CommitBorderToGlobalMemory(mln::bp::Tile2DView<int> aux, int bx, int by)
{
  int w = m_data->nodemap.width();
  int h = m_data->nodemap.height();
  int x = bx * (Executor::MAX_TILE_WIDTH - 1);
  int y = by * (Executor::MAX_TILE_HEIGHT - 1);
  int width = std::min(Executor::MAX_TILE_WIDTH, w - x);
  int height = std::min(Executor::MAX_TILE_HEIGHT, h - y);

  int ty = by * Executor::MAX_TILE_HEIGHT;
  int tx = bx * Executor::MAX_TILE_WIDTH;


  auto nodemap = aux.clip(1, 1, width, height);

  // Copy upper border
  if (by > 0)
    std::copy_n(nodemap.row(0), width, m_data->halo_rows.row(2 * by - 1) + tx);

  // Copy lower border
  if (by < (m_data->ny - 1))
    std::copy_n(nodemap.row(height - 1), width, m_data->halo_rows.row(2 * by) + tx);

  // Copy left column
  if (bx > 0)
  {
    auto left = m_data->halo_cols.row(2 * bx - 1) + ty;
    for (int y = 0; y < height; ++y)
      left[y] = nodemap(0, y);
  }

  // Copy right column
  if (bx < (m_data->nx - 1))
  {
    auto right = m_data->halo_cols.row(2 * bx) + ty;
    for (int y = 0; y < height; ++y)
      right[y] = nodemap(width-1, y);
  }
}

// Canocalize parent
void Executor::Impl::canonicalize_parent_and_nodemap(int bx, int by)
{
  int TILE_SIZE = Executor::MAX_TILE_WIDTH * Executor::MAX_TILE_HEIGHT;
  int w         = m_data->nodemap.width();
  int h         = m_data->nodemap.height();
  int x         = bx * (Executor::MAX_TILE_WIDTH - 1);
  int y         = by * (Executor::MAX_TILE_HEIGHT - 1);
  int width     = (bx != (m_data->nx - 1)) ? (Executor::MAX_TILE_WIDTH - 1) : (w - x);  // Skip the right halo
  int height    = (by != (m_data->ny - 1)) ? (Executor::MAX_TILE_HEIGHT - 1) : (h - y); // Skip the bottom halo

  // Ensure the parent[x] is canonical for each node x of the tile
  {
    int start = by * m_data->nx * TILE_SIZE  + bx * TILE_SIZE;
    canonicalize_parent(m_data->parent, m_data->levels, start, start + TILE_SIZE);
  }

  // Ensure nodemap[x] is canonical
  {
    auto nodemap = m_data->nodemap.clip(x, y, width, height);
    canonicalize_nodemap(nodemap, m_data->parent, m_data->levels);
  }
}


// Canocalize parent
void Executor::Impl::canonicalize_parent_and_nodemap_auto(int bx, int by)
{
  int kTileSize = Executor::MAX_TILE_WIDTH * Executor::MAX_TILE_HEIGHT;
  int w         = m_data->nodemap.width();
  int h         = m_data->nodemap.height();
  int x         = bx * (Executor::MAX_TILE_WIDTH - 1);
  int y         = by * (Executor::MAX_TILE_HEIGHT - 1);

  int global_id = by * m_data->nx * kTileSize  + bx * kTileSize;
  // Ensure the parent[x] is canonical for each node x of the tile
  {
    int width = std::min(Executor::MAX_TILE_WIDTH, w - x);
    int height = std::min(Executor::MAX_TILE_HEIGHT, h - y);
    canonicalize_parent(m_data->parent, m_data->levels, width, height, global_id, Executor::MAX_TILE_WIDTH);
  }

  // Ensure nodemap[x] is canonical
  {
    int width     = (bx != (m_data->nx - 1)) ? (Executor::MAX_TILE_WIDTH - 1) : (w - x);  // Skip the right halo
    int height    = (by != (m_data->ny - 1)) ? (Executor::MAX_TILE_HEIGHT - 1) : (h - y); // Skip the bottom halo
    auto nodemap = m_data->nodemap.clip(x, y, width, height);
    canonicalize_nodemap_autoindex(nodemap, m_data->parent, m_data->levels, global_id, Executor::MAX_TILE_WIDTH);
  }
}

