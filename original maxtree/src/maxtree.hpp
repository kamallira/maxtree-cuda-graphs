#pragma once

#include <mln/bp/tile.hpp>

/******************************************/
/****       Maxtree computation       ****/
/******************************************/

// Note: this function is non-allocating, aux data must be alloacted before
// Nodemap must be allocated with a border of size 1 and initialized with INT32_MAX
// Nodemap inter value values must be -1
std::size_t maxtree2d_salembier(mln::bp::Tile2DView<uint8_t> input,        //
                                mln::bp::Tile2DView<int>     node_map,     //
                                int*                         parent,       //
                                uint8_t*                     levels,       //
                                int                          connectivity, //
                                std::size_t                  count = 0);


// Compute the maxtree (1D) of each line of the tile
void compute_maxtree_hor_1d(mln::bp::Tile2DView<uint8_t> input,           //
                            mln::bp::Tile2DView<int>     parent,          //
                            int                          global_id   = 0, //
                            int                          index_pitch = 0);


//*****************************************/
/****         Merging routines         ****/
/******************************************/


void merge_halo(mln::bp::Tile2DView<int> A_nodemap, //
                mln::bp::Tile2DView<int> B_nodemap, //
                int*                     parent,    //
                uint8_t*                 levels,    //
                int                      axis);

void merge_halo(int*        A_nodemap, //
                int*        B_nodemap, //
                std::size_t n,         //
                int*        parent,    //
                uint8_t*    levels);

void merge_halo(int         gid_1,  //
                int         gid_2,  //
                std::size_t n,      //
                int*        parent, //
                uint8_t*    levels, //
                int         step);


void merge_rows(int* __restrict parent, uint8_t* __restrict levels, int width, int height, int global_id,
                int index_pitch, int connectivity);


/******************************************/
/****         Misc routines            ****/
/******************************************/

void canonicalize(mln::bp::Tile2DView<int> nodemap, int* __restrict parent, uint8_t* __restrict levels, std::size_t n);


///
///  \brief Canonicalize the parent array
///
///  The parent array is processed until the sentinel value is encountered (MIN_INT32) or the end is reached
///  After canonicalization,  ∀ x, parent[x] is a canonical element.
///
/// A canonical element satifies either:
/// * parent[x] = -1 (root)
/// * levels[parent[x]] != levels[x]
void canonicalize_parent(int* __restrict parent, uint8_t* __restrict levels, std::size_t begin, std::size_t end);
void canonicalize_parent(int* parent, uint8_t* levels, int width, int height, int global_id, int pitch_index);

void canonicalize_nodemap(mln::bp::Tile2DView<int> nodemap, int* __restrict parent, uint8_t* __restrict levels);

void canonicalize_nodemap_autoindex(mln::bp::Tile2DView<int> nodemap, int* __restrict parent,
                                    uint8_t* __restrict levels, int global_index, int index_pitch);
