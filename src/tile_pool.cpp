#include "tile_pool.hpp"
#include <mutex>
#include <cassert>

TileMemoryPool::TileMemoryPool(int width, int height)
  : m_width{width},
    m_height{height}
{
}

void* TileMemoryPool::_acquire()
{
  // Consume
  std::lock_guard<std::mutex> lk(m_mutex);
  if (m_size == 0)
    return this->_allocate();
  return m_memory_pool[--m_size];
}


void TileMemoryPool::_release(void* buffer)
{
  // Producer
  if (buffer)
  {
    std::lock_guard<std::mutex> lk(m_mutex);
    assert(m_size < m_capacity);
    m_memory_pool[m_size++] = buffer;
  }
}



