#pragma once
#include <cstdint>


template <class T>
[[gnu::always_inline]] __device__ inline T* ptr_offset(T* in, std::ptrdiff_t offset)
{
  return (T*)((char*)in + offset);
}


template <class T>
class Tile2DView
{
public:
  Tile2DView() = default;

  __device__ Tile2DView(T* ptr, int width, int height, int pitch) noexcept
    : m_ptr{ptr}
    , m_width{width}
    , m_height{height}
    , m_pitch{pitch}
    {
  }

  /// \brief Return a pointer to the beginning of the row
  __device__ T* row(int y) const noexcept { return m_ptr + y * m_pitch; }


  /// \brief Return a raw to pointer to the buffer
  __device__ T* data() const noexcept { return m_ptr; }

  /// \brief Get the stride (in **number of elements**)
  __device__ int pitch() const noexcept { return m_pitch; }

  /// \Brief Get the number of cols
  __device__ int width() const noexcept { return m_width; }

  /// \brief Get the number of rows
  __device__ int height() const noexcept { return m_height; }

  /// \brief Return true if the buffer aligned with the required size
  /// e.g., ``is_aligned(16)`` to check that the buffer is 16-bytes aligned
  __device__ bool is_aligned(int width = 16) { return ((intptr_t)m_ptr & (intptr_t)(width - 1)) == 0; }

  __device__ T& at(int x, int y) const noexcept { return m_ptr[y * m_pitch + x]; }

  __device__ T& operator()(int x, int y) const noexcept { return m_ptr[y * m_pitch + x]; }

  __device__ T& operator[](int index) const noexcept { return m_ptr[index]; }

protected:
  T*  m_ptr = nullptr;
  int m_width;
  int m_height;
  int m_pitch;
};
