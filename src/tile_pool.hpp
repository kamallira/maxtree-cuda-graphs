#pragma once
#include <mln/bp/tile.hpp>
#include <mutex>


class TileMemoryPool
{
public:
  TileMemoryPool(int width, int height);
  virtual ~TileMemoryPool() = default;

  TileMemoryPool(const TileMemoryPool&) = delete;
  TileMemoryPool& operator=(const TileMemoryPool&) = delete;

protected:
  void*         _acquire();
  void          _release(void*);
  virtual void* _allocate() = 0;

  int              m_capacity = 64;
  int              m_width;
  int              m_height;
  std::ptrdiff_t   m_stride = 0;

  std::mutex 	   m_mutex;
  int              m_size = 0;
  void*            m_memory_pool[64];
};


template <class T>
class TileMemoryPoolT : public TileMemoryPool
{
public:
  using TileMemoryPool::TileMemoryPool;

  ~TileMemoryPoolT();
  void acquire(mln::bp::Tile2D<T>& tile);
  void release(mln::bp::Tile2D<T>& tile);
private:
  void* _allocate() final;
};



/******************************************/
/****          Implementation          ****/
/******************************************/


template <class T>
TileMemoryPoolT<T>::~TileMemoryPoolT()
{
  for (int i = 0; i < this->m_size; ++i)
    mln::bp::aligned_free_2d<T>((T*) this->m_memory_pool[i], this->m_width, this->m_height, this->m_stride);
}


template <class T>
void* TileMemoryPoolT<T>::_allocate()
{
  std::ptrdiff_t pitch;
  T* buffer = mln::bp::aligned_alloc_2d<T>(m_width, m_height, pitch);
  if (m_stride == 0)
    m_stride = pitch;
  return buffer;
}


template <class T>
void TileMemoryPoolT<T>::acquire(mln::bp::Tile2D<T>& tile)
{
  void* buffer = TileMemoryPool::_acquire();
  tile = mln::bp::Tile2D<T>::acquire((int*)buffer, m_width, m_height, m_stride);
}


template <class T>
void TileMemoryPoolT<T>::release(mln::bp::Tile2D<T>& tile)
{
  TileMemoryPool::_release(tile.release());
}

