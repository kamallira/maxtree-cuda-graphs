#include "histogram2d.hpp"


#include <mln/bp/utils.hpp>
#include <mln/core/image/image.hpp>
#include <mln/core/point.hpp>
#include <mln/morpho/private/pqueue.hpp>

#include <vector>
#include <cassert>

using node_id_t = int;

struct root_info_t
{
  node_id_t root_id;
  int       level;
};


class LevelRootStack
{
public:
  LevelRootStack() = default;

  void push(root_info_t x)
  {
    assert(m_size <= UINT8_MAX);
    m_roots[m_size++] = x;
  }
  root_info_t pop()
  {
    assert(m_size > 0);
    return m_roots[--m_size];
  }
  root_info_t top() const
  {
    assert(m_size > 0);
    return m_roots[m_size - 1];
  }
  bool empty() const { return m_size == 0; }

private:
  root_info_t m_roots[UINT8_MAX + 1];
  int         m_size = 0;
};

enum st : int32_t
{
  NONE    = -1,
  INQUEUE = 0,
  DONE    = INT32_MAX
};

//
// input: original image
//
std::size_t maxtree2d_salembier(mln::bp::Tile2DView<uint8_t> input, //
                                mln::bp::Tile2DView<int>     node_map,
                                node_id_t*                   parent,       //
                                uint8_t*                     levels,       //
                                int                          connectivity, //
                                std::size_t                  count)
{
  using small_point2d = mln::ndpoint<2, short>;

  GSL_ASSUME(connectivity == 4 || connectivity == 8);

  // Hierarchical queue
  using queue_impl_t = mln::morpho::details::hvectors<small_point2d>;
  using queue_t      = mln::morpho::detail::hpqueue<256, small_point2d, queue_impl_t, false>;

  queue_t        queue(input);
  LevelRootStack roots;

  std::array<mln::point2d, 8> nbh = {{{0, -1}, {-1, 0}, {1, 0}, {0, 1}, {-1, -1}, {1, -1}, {-1, 1}, {1, 1}}};


  mln::point2d p             = {0, 0};
  int          current_level = input(p.x(), p.y());
  int          pstatus;
  queue.push_last(current_level, p);

  // Start flooding
start_flooding:
  parent[count] = -1;
  levels[count] = current_level;
  roots.push({(node_id_t)count, current_level});
  pstatus = INQUEUE;
  count++;

keep_flooding :

{
  for (int k = 0; k < connectivity; ++k)
  {
    int          mask = 1 << k;
    mln::point2d n    = p + nbh[k];

    if ((pstatus & mask) || node_map(n.x(), n.y()) != NONE)
      continue;

    // Insert n INQUEUE
    auto nval              = input(n.x(), n.y());
    node_map(n.x(), n.y()) = INQUEUE;
    queue.push_last(nval, n);
    pstatus |= mask;

    // If the neighbor is lower, postpone the neighbor
    if (nval <= current_level)
      continue;

    // Otherwise, process it, (do not remove p from stack)
    node_map(p.x(), p.y()) = pstatus;
    current_level          = nval;
    p                      = n;
    goto start_flooding;
  }
}

  // All the neighbors have been seen, p is DONE
  // status(p) becomes >= 0
  node_map(p.x(), p.y()) = roots.top().root_id;
  queue.pop();

  // If the queue gets empty, we have processed the whole image
  if (queue.empty())
    goto end_flooding;

  {
    auto old_level             = current_level;
    std::tie(current_level, p) = queue.top();
    pstatus                    = node_map(p.x(), p.y());
    if (current_level == old_level)
      goto keep_flooding;
    // HOOK: the flood has endend
    // Attach to the parent
    {
      auto new_level = current_level;

      // Attach to parent
      assert(!roots.empty());
      auto current = roots.top();
      roots.pop();

      assert(old_level == current.level);
      assert(old_level > new_level);
      // Fixme: optimize this test with a sentinel value

      root_info_t par;
      if (!roots.empty())
        par = roots.top();

      if (roots.empty() || par.level != new_level)
      {
        parent[count] = -1;
        levels[count] = new_level;
        par           = {(node_id_t)count, new_level};
        roots.push(par);
        count++;
      }
      assert(par.level <= new_level);

      parent[current.root_id] = par.root_id;
    }
  }

  // In the max-tree case, we just keep flooding
  if (true)
    goto keep_flooding;
  else
    goto start_flooding;

  // End: there is no more point to process
end_flooding:
  // HOOK: the flood has endend
  // Attach to the parent
  {
    node_id_t root = roots.top().root_id;
    roots.pop();

    assert(roots.empty());
    parent[root] = -1;
  }
  return count;
}
