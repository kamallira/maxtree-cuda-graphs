#pragma once
#include <cstdint>
#include "tile.cuh"
#include <bit>
#include <tuple>
#include "maxtree_defs.hpp"

#include <cuda/std/utility>

#define ENABLE_KERNEL_TIMERS 0

__device__ unsigned long long timers[5];



struct __align__(4) info_t 
{

#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__

  uint8_t   level;
  uint8_t   parent_level;
  int16_t   parent_index;


#elif __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__

  int16_t   parent_index;   // Bits 15 - 0
  uint8_t   parent_level;   // Bits 23 - 16
  uint8_t   level;          // Bits 31 - 24


  static constexpr int      get_par_index(uint32_t x) { return x & 0x0000FFFF; }
  static constexpr int      get_par_level(uint32_t x) { return (x & 0x00FF0000) >> 16; }
  static constexpr int      get_level(uint32_t x) { return (x & 0xFF000000) >> 24; }
  static constexpr bool     isRoot(uint32_t x) { return (x & 0x0000FFFF) == kRootParentIndex; }
  static constexpr uint32_t make(int level, int parent_level, int parent_index)
  {
    return ((uint32_t)level << 24) | ((uint32_t)parent_level << 16) | (uint32_t)parent_index;
  }

#endif

  static inline constexpr int16_t kRootParentIndex = INT16_MAX;
  static inline constexpr uint8_t kRootParentLevel = UINT8_MAX;

  __device__ constexpr bool                     is_root() noexcept { return this->parent_index == kRootParentIndex; }
  __device__ constexpr uint32_t                 as_uint32() const noexcept { return *(uint32_t*)(this); }
  __device__ static constexpr info_t            from_uint32(int32_t x) noexcept { return *(info_t*)(&x); }




  __device__ constexpr bool operator==(info_t other) const noexcept { return this->as_uint32() == other.as_uint32(); }
  __device__ constexpr bool operator!=(info_t other) const noexcept { return this->as_uint32() != other.as_uint32(); }
  __device__ constexpr bool operator<=(info_t other) const noexcept { return this->as_uint32() <= other.as_uint32(); }
  __device__ constexpr bool operator>=(info_t other) const noexcept { return this->as_uint32() >= other.as_uint32(); }
  __device__ constexpr bool operator<(info_t other)  const noexcept { return this->as_uint32() <  other.as_uint32(); }
  __device__ constexpr bool operator>(info_t other)  const noexcept { return this->as_uint32() >  other.as_uint32(); }

  __device__ void swap(info_t& other) noexcept { cuda::std::swap(*(uint32_t*)this, *(uint32_t*)(&other)); }
};

__device__
void swap(info_t& a, info_t& b) noexcept
{
  a.swap(b);
}

static_assert(sizeof(info_t) == sizeof(uint32_t));




__device__ void compute_maxtree_uf(info_t* parent, int pitch, int connectivity);
__device__ int flatten(uint32_t* tile, int pitch, int x, int y);
__device__ void flatten_column(uint32_t* tile, int pitch, int n);


// Wrapper function for host
__device__ void compute_maxtree_tile(const uint8_t* __restrict__ input_, int32_t* __restrict__ parent_,
                                     info_t* __restrict__ aux_, int width, int height, int pitch, int tile_width,
                                     int tile_height, int y_start_index, uint32_t variant);


namespace
{
  template <class T, class U>
  __device__ __forceinline__
  T bit_cast(U x)
  {
    static_assert(sizeof(T) == sizeof(U));
    return *(T*)(&x);
  }
}
