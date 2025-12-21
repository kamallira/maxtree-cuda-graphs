#include <mln/bp/tile.hpp>
#include <mln/bp/functional.hpp>

namespace
{

  class LevelRootStack
  {
  public:
    LevelRootStack() = default;

    void push(int x)
    {
      assert(m_size <= UINT8_MAX);
      m_roots[m_size++] = x;
    }
    int pop()
    {
      assert(m_size > 0);
      return m_roots[--m_size];
    }
    int top() const
    {
      assert(m_size > 0);
      return m_roots[m_size - 1];
    }
    bool empty() const { return m_size == 0; }

  private:
    int         m_roots[UINT8_MAX+1];
    int         m_size = 0;
  };


  // Unstack and attach the parent until the stack get empty or the stack level <= v
  int unstack(const uint8_t* __restrict input, int* __restrict parent, LevelRootStack& stack, int v)
  {
    assert(!stack.empty());

    int r = stack.pop();
    while (!stack.empty() && input[stack.top()] > v)
      r = (parent[r] = stack.pop());

    assert(stack.empty() || input[stack.top()] <= v);
    return r;
  }

  template <class T>
  int compute_maxtree_1d(const uint8_t* __restrict input, T* __restrict parent, std::size_t n)
  {
    LevelRootStack roots;
    roots.push(0);

    for (std::size_t i = 1; i < n; ++i)
    {
      auto v = input[i];
      auto top = input[roots.top()];

      if (top < v)
      {
        roots.push(i);
      }
      else if (top == v)
      {
        parent[i] = roots.top();
      }
      else
      {
        int r = unstack(input, parent, roots, v);

        if (roots.empty() || input[roots.top()] < v)
        {
          parent[r] = i;
          roots.push(i);
        }
        else
        {
          parent[r] = (parent[i] = roots.top());
        }
      }
    }
    int root = unstack(input, parent, roots, INT32_MIN);
    parent[root] = -1;

    return root;
  }
}

void compute_maxtree_hor_1d(mln::bp::Tile2DView<uint8_t> input, //
                            mln::bp::Tile2DView<int> parent, int global_id, int index_pitch)
{
  // int x = item.get_global_id(0);
  // int y0 = item.get_global_id(1) * tile_height;
  // int y1 = std::min(y0 + tile_height, input_.height());
  int width  = input.width();
  int height = input.height();

  for (int y = 0; y < height; ++y)
  {
    auto input_line  = input.row(y);
    auto parent_line = parent.row(y);
    int root = compute_maxtree_1d(input_line, parent_line, width);
    mln::bp::apply(parent_line, width, [off = global_id + y * index_pitch](auto v) { return v + off; });
    parent_line[root] = -1;
  }
}
