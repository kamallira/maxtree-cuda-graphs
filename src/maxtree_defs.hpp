#pragma once

enum algo_variation
{
  GRID_CONNECTION_OPTIM_MASK = 0xF00, // 1111 0000 0000
  GRID_CONNECTION_OPTIM_ON   = 0x100, // 0001 0000 0000
  GRID_CONNECTION_OPTIM_OFF  = 0x000, // 0001 0000 0000
  GRID_HALO_ON               = 0x200, // 0010 0000 0000
  MAXTREE_ALGORITH_MASK      = 0xF0,  // 0000 1111 0000
  MAXTREE_BASE               = 0x10,  // 0000 0001 0000
  MAXTREE_OPTIM_1D           = 0x20,  // 0000 0010 0000
  MAXTREE_CONNECTIVITY_MASK  = 0x0F,  // 0000 0000 1111
  MAXTREE_C4                 = 0x04,  // 0000 0000 0010
  MAXTREE_C8                 = 0x08,  // 0000 0000 0100
};


class Chrono
{
public:
  virtual void PauseTiming()                    = 0;
  virtual void ResumeTiming()                   = 0;
  virtual void SetIterationTime(double seconds) = 0;
};
