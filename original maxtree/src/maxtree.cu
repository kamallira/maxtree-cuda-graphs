#include <fmt/core.h>
#include "maxtree.cuh"
#include <cuda_profiler_api.h>

namespace
{

  template <class... TArgs>
  void checkError(cudaError_t err, const char* msg, TArgs&&... args)
  {
    if (err != cudaSuccess)
    {
      fmt::print(stderr, msg, std::forward<TArgs>(args)...);
      fmt::print(stderr, "\nCUDA msg: {}\n", cudaGetErrorString(err));
    }
  }
}



void maxtree_cu(const uint8_t* input_buffer, std::ptrdiff_t input_stride, int width, int height, int32_t* parent, std::ptrdiff_t parent_stride, uint32_t variant, Chrono* chrono)
{
  const int connectivity = variant & MAXTREE_CONNECTIVITY_MASK;
  assert(connectivity == 4 || connectivity == 8);

  constexpr int TILE_WIDTH  = 64;
  constexpr int TILE_HEIGHT = 16;

  cudaProfilerStart();
  cudaFuncSetCacheConfig(compute_maxtree_tile_base_optim<TILE_WIDTH, TILE_HEIGHT>, cudaFuncCachePreferShared);
  cudaFuncSetCacheConfig(compute_maxtree_tile_optim_1d_connection<TILE_WIDTH, TILE_HEIGHT>, cudaFuncCachePreferShared);
  cudaFuncSetCacheConfig(compute_maxtree_tile_base<TILE_WIDTH, TILE_HEIGHT>, cudaFuncCachePreferShared);
  cudaFuncSetCacheConfig(compute_maxtree_tile_optim_1d<TILE_WIDTH, TILE_HEIGHT>, cudaFuncCachePreferShared);
  cudaFuncSetCacheConfig(merge_maxtree_v<uint8_t>, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(merge_maxtree_v_optim, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(merge_maxtree_h<uint8_t>, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(merge_maxtree_h_optim, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig((void(*)(const uint8_t*, int32_t*, int, int, int, int)) flatten, cudaFuncCachePreferL1);


  uint8_t* dInput;
  int32_t* dParent;
  int      pitch;

  cudaError_t err;

  {
    std::size_t dInputPitch, dParentPitch;
    err = cudaMallocPitch(&dInput, &dInputPitch, width * sizeof(uint8_t), height);
    checkError(err, "Unable to allocate Input Image memory\n");
    err = cudaMallocPitch(&dParent, &dParentPitch, dInputPitch * sizeof(int32_t), height);
    checkError(err, "Unable to allocate Parent memory\n");
    assert(dParentPitch == (dInputPitch * sizeof(int32_t)));

    // number of elements (=bytes for input) between two lines
    pitch = dInputPitch;
  }


  constexpr int kStreamSplit = 4;
  cudaStream_t kProcessStream;
  cudaStream_t kMemoryStream;
  cudaStreamCreate(&kProcessStream);
  cudaStreamCreate(&kMemoryStream);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int count_tile_x = (width + TILE_WIDTH - 1) / TILE_WIDTH;
  int count_tile_y = (height + TILE_HEIGHT - 1) / TILE_HEIGHT;
  int kGridRowPerStream = std::max(1, (count_tile_y + kStreamSplit - 1) / kStreamSplit);

  unsigned long long _timers[5] = {0};
  if constexpr (ENABLE_KERNEL_TIMERS)
  {
    err = cudaMemcpyToSymbol(timers, _timers, sizeof(timers), 0, cudaMemcpyHostToDevice);
    checkError(err, "Unable to memset");
  }


  /////////////////////////////////////////////////////////////////////////////////
  //////////////////           LOCAL MAXTREE COMPUTATION       ////////////////////
  /////////////////////////////////////////////////////////////////////////////////
  for (int ystart = 0; ystart < height; ystart += TILE_HEIGHT * kGridRowPerStream)
  {
    int nRows     = std::min(height - ystart, kGridRowPerStream * TILE_HEIGHT);
    int nGridRows = (nRows + TILE_HEIGHT - 1) / TILE_HEIGHT;

    // Enqueue copy on stream 1
    {
      err   = cudaMemcpy2DAsync(dInput + ystart * pitch, pitch * sizeof(uint8_t),   //
                                input_buffer + ystart * input_stride, input_stride, //
                                width * sizeof(uint8_t), nRows, cudaMemcpyHostToDevice, kMemoryStream);
      checkError(err, "Unable to copy data with nrows = {}", nRows);
    }

    // If kernel benchmark, sync and start chrono
    cudaStreamSynchronize(kMemoryStream);

    if (ystart == 0)
      cudaEventRecord(start);

    // Enqueue process on stream 2
    if ((variant & MAXTREE_ALGORITH_MASK) == MAXTREE_OPTIM_1D)
    {
      dim3 threads_per_block(TILE_WIDTH, 1);
      dim3 grid_size(count_tile_x, nGridRows);
      if ((variant & GRID_CONNECTION_OPTIM_MASK) == GRID_CONNECTION_OPTIM_ON)
      {
        compute_maxtree_tile_optim_1d_connection<TILE_WIDTH, TILE_HEIGHT>
        <<<grid_size, threads_per_block, 0, kProcessStream>>>(dInput, dParent, width, height, pitch, ystart, connectivity);
      }
      else
      {
        compute_maxtree_tile_optim_1d<TILE_WIDTH, TILE_HEIGHT>
        <<<grid_size, threads_per_block, 0, kProcessStream>>>(dInput, dParent, width, height, pitch, ystart, connectivity);
      }
      err = cudaGetLastError();
      checkError(err, "Unable to execute the kernel");
    }
    else if ((variant & MAXTREE_ALGORITH_MASK) == MAXTREE_BASE)
    {
      dim3 threads_per_block(TILE_WIDTH, TILE_HEIGHT);
      dim3 grid_size(count_tile_x, nGridRows);
      if ((variant & GRID_CONNECTION_OPTIM_MASK) == GRID_CONNECTION_OPTIM_ON)
      {
        compute_maxtree_tile_base_optim<TILE_WIDTH, TILE_HEIGHT>
        <<<grid_size, threads_per_block, 0, kProcessStream>>>(dInput, dParent, width, height, pitch, ystart, connectivity);
      }
      else
      {
        compute_maxtree_tile_base<TILE_WIDTH, TILE_HEIGHT>
        <<<grid_size, threads_per_block, 0, kProcessStream>>>(dInput, dParent, width, height, pitch, ystart, connectivity);
      }
      err = cudaGetLastError();
      checkError(err, "Unable to execute the kernel");
    }
  }

  if constexpr (ENABLE_KERNEL_TIMERS)
  {
    err = cudaMemcpyFromSymbol(_timers, timers, sizeof(_timers), 0, cudaMemcpyDeviceToHost);
    checkError(err, "Unable to memcpy symbol");
    printf("Timers %i = %llu\n", 0, _timers[0]);
    printf("Timers %i = %llu\n", 1, _timers[1]);
    printf("Timers %i = %llu\n", 2, _timers[2]);
    printf("Timers %i = %llu\n", 3, _timers[3]);
    printf("Timers %i = %llu\n", 4, _timers[4]);
  }

  /////////////////////////////////////////////////////////////////////////////////
  //////////////////        LOCAL MAXTREEE MERGING     ////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////


  if (count_tile_x > 1)
  {
    constexpr int NTHREAD_MERGE_X = 32;
    constexpr int NTHREAD_MERGE_Y = 32;
    int           nbloc_y         = (height + NTHREAD_MERGE_Y - 1) / NTHREAD_MERGE_Y;
    int           nbloc_x         = (count_tile_x - 1 + NTHREAD_MERGE_X - 1) / NTHREAD_MERGE_X;

    dim3 threads_per_block(NTHREAD_MERGE_X, NTHREAD_MERGE_Y);
    dim3 grid_size(nbloc_x, nbloc_y);

    if ((variant & GRID_CONNECTION_OPTIM_MASK) == GRID_CONNECTION_OPTIM_ON)
      merge_maxtree_h_optim<<<grid_size, threads_per_block>>>(dInput, dParent, pitch, width, height, TILE_WIDTH, TILE_HEIGHT, connectivity);
    else
      merge_maxtree_h<<<grid_size, threads_per_block>>>(dInput, dParent, pitch, width, height, TILE_WIDTH, connectivity);
    err = cudaGetLastError();
    checkError(err, "Unable to execute the kernel MERGE");
  }


  if (count_tile_y > 1)
  {
    constexpr int NTHREAD_MERGE_X = 256;
    constexpr int NTHREAD_MERGE_Y = 4;
    int           nbloc_y         = (count_tile_y - 1 + NTHREAD_MERGE_Y - 1) / NTHREAD_MERGE_Y;
    int           nbloc_x         = (width + NTHREAD_MERGE_X - 1) / NTHREAD_MERGE_X;

    dim3 threads_per_block(NTHREAD_MERGE_X, NTHREAD_MERGE_Y);
    dim3 grid_size(nbloc_x, nbloc_y);
    
    if ((variant & GRID_CONNECTION_OPTIM_MASK) == GRID_CONNECTION_OPTIM_ON)
      merge_maxtree_v_optim<<<grid_size, threads_per_block>>>(dInput, dParent, pitch, width, height, TILE_HEIGHT, connectivity);
    else
      merge_maxtree_v<<<grid_size, threads_per_block>>>(dInput, dParent, pitch, width, height, TILE_HEIGHT, connectivity);
    err = cudaGetLastError();
    checkError(err, "Unable to execute the kernel MERGE");
  }



  assert(parent_stride == pitch * sizeof(int32_t));

  /////////////////////////////////////////////////////////////////////////////////
  //////////////////        FLATTENING + COMMIT GLOBAL   ////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////


  {
    constexpr int TILE_WIDTH  = 32;
    constexpr int TILE_HEIGHT = 32;
    const int count_tile_x = (width + TILE_WIDTH - 1) / TILE_WIDTH;
    const int count_tile_y = (height + TILE_HEIGHT - 1) / TILE_HEIGHT;
    int kGridRowPerStream = std::max(1, (count_tile_y + kStreamSplit - 1 ) / kStreamSplit);

    // So we compute on stream 0 and copy on stream 1
    for (int ystart = 0; ystart < height; ystart += kGridRowPerStream * TILE_HEIGHT)
    {
      int nRows     = std::min(height - ystart, kGridRowPerStream * TILE_HEIGHT);
      int nGridRows = (nRows + TILE_HEIGHT - 1) / TILE_HEIGHT;

      dim3 threads_per_block(TILE_WIDTH, TILE_HEIGHT);
      dim3 grid_size(count_tile_x, nGridRows);

      // Enqueue process on stream 0
      {
        flatten<<<grid_size, threads_per_block, 0, kProcessStream>>>(dInput, dParent, width, height, pitch, ystart);
        err = cudaGetLastError();
        checkError(err, "Unable to execute the kernel FLATTEN");
      }

      cudaStreamSynchronize(kProcessStream);

      // Enqueue copy on stream 1
      {
        err = cudaMemcpy2DAsync(parent + ystart * pitch, parent_stride,            //
                                dParent + ystart * pitch, pitch * sizeof(int32_t), //
                                width * sizeof(int32_t), nRows, cudaMemcpyDeviceToHost, kMemoryStream);
        checkError(err, "Unable to copy back data");
      }
    }

    // If kernel benchmark stop chrono
    cudaEventRecord(stop, kProcessStream);
  }

  cudaFree(dInput);
  cudaFree(dParent);
  cudaStreamDestroy(kMemoryStream);
  cudaStreamDestroy(kProcessStream);
  cudaDeviceSynchronize();
  cudaProfilerStop();
  if (chrono)
  {
    float duration_ms = 0;
    cudaEventElapsedTime(&duration_ms, start, stop);
    chrono->SetIterationTime(duration_ms / 1000.);
  }
}


namespace cuda
{
  int cudaHostRegister(void* ptr, size_t size, unsigned int flags)
  {
    auto err = ::cudaHostRegister(ptr, size, flags);
    checkError(err, "Unable to pin memory");
    return err;
  }

  int cudaHostUnregister(void* ptr)
  {
    return ::cudaHostUnregister(ptr);
  }

  void cudaFreeHost(void* ptr)
  {
    auto err = ::cudaFreeHost(ptr);
    checkError(err, "Unable to release memory");
  }
}

void maxtree_allocate_output(int32_t** ptr, int width, int height, int size, int* pitch)
{
  constexpr uint32_t a         = 512;
  int                linebytes = (width * size + a - 1) & ~(a - 1);
  int                linesize  = linebytes / size;

  auto err = cudaMallocHost(ptr, linesize * sizeof(int32_t) * height);
  checkError(err, "Unable to allocate HOST output");
  *pitch = linesize;
}
