#include "executor.hpp"
#include "tile_pool.hpp"
#include <mln/bp/tile.hpp>
#include <span>


/*
class MaxtreeExecutorImpl : public Executor::ImplBase
{
public:
  // Constructor
  MaxtreeExecutorImpl(MaxtreeData* mt, TileMemoryPoolT<int>* pool);
  ~MaxtreeExecutorImpl() final;


  // Copy the executor
  std::unique_ptr<Executor::ImplBase> Clone() final;

  // Apply step on the block (bx, by) using Salembier algorithm
  void ComputeLocalMaxtree_Salembier(int bx, int by, int connectivity);

  // Apply step on the block (bx, by) using 1D algorithm
  void ComputeLocalMaxtree_1D(int bx, int by, int connectivity);



  // Merge along a bottom or right maxtrees of the given tile
  // This method does not support merging concurrently on the same areas
  // A correction reduction must be performed
  void MergeMaxtrees(int bx, int by, int nbloc, int axis) final;

  // Canocalize parent & nodemap
  void Canonicalize(int bx, int by) final;


private:
  // Copy the halo zone from local memory to the global memory
  void CommitBorderToGlobalMemory(mln::bp::Tile2DView<int> aux, int bx, int by);
  void CommitTileToGlobalMemory(mln::bp::Tile2DView<int> aux, int bx, int by);

private:
  // Local tiles attributes
  //mln::bp::Tile2D<int>         m_aux; // Aux data for nodemap computation
};
*/
