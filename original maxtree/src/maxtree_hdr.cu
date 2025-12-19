#include <cassert>
#include <cstdio>
#include <cuda/std/utility>
#include <optional>
#include <utility>
//#include <cuda/std/optional>
#include <cooperative_groups.h>


namespace
{

  template <int TILE_WIDTH, int TILE_HEIGHT>
  __global__ void compute_maxtree_tile_HDR(const uint16_t* __restrict__ input, int32_t* __restrict__ parent,
                                           int width, int height, int pitch,
                                           int y_start_index, int connectivity)
  {
    assert(connectivity == 4 || connectivity == 8);
    __shared__ uint16_t sInput[TILE_HEIGHT * TILE_WIDTH];
    __shared__ int32_t  sParent[TILE_HEIGHT * TILE_WIDTH];

    int bx   = blockIdx.x * TILE_WIDTH;
    int by   = y_start_index + blockIdx.y * TILE_HEIGHT;
    int gx   = bx + threadIdx.x;
    int gy   = by + threadIdx.y;
    int w    = std::min(width - bx, TILE_WIDTH);
    int h    = std::min(height - by, TILE_HEIGHT);


    int gidx  = gy * pitch + gx;

    int a = threadIdx.y * TILE_WIDTH + threadIdx.x;

    // 1. Copy to shared memory
    if (threadIdx.x < w && threadIdx.y < h)
    {
      sInput[a]  = input[gidx];
      sParent[a] = -1;
    }

    __syncthreads();

    // 2. Compute maxtree of the tile
    if (threadIdx.x < w && threadIdx.y < h)
    {
      if (threadIdx.x > 0)
      {
        int b = a - 1;
        if (connectivity == 8 && threadIdx.y > 0)
          uf_zip_global(sInput, sParent, a - TILE_WIDTH, b);

        uf_zip_global(sInput, sParent, a, b);

        if (connectivity == 8 && threadIdx.y > 0)
          uf_zip_global(sInput, sParent, a, b - TILE_WIDTH);
      }

      if (threadIdx.y > 0)
        uf_zip_global(sInput, sParent, a, a - TILE_WIDTH);
    }

    __syncthreads();

    // Flatten Levels & Commit
    if (threadIdx.x < w && threadIdx.y < h)
    {
      int qindex = sParent[a];
      if (qindex != -1)
        qindex = uf_find_peak_root(sInput, sParent, qindex, sInput[qindex], nullptr);

      cooperative_groups::coalesced_threads().sync();
      sParent[a] = qindex;

      int qy       = qindex / TILE_WIDTH;
      int qx       = qindex % TILE_WIDTH;
      assert(qy < h && qx < w);
      int qgidx    = (by + qy) * pitch + (bx + qx);
      parent[gidx] = (qindex == -1) ? -1 : qgidx;
    }

    __syncthreads();

    // Flatten Buckets & Commit
    /*
    if (threadIdx.x < w && threadIdx.y < h)
    {
      int qindex       = sParent[a];
      int bucket_value = (sInput[qindex] | 0x00FF) + 0x0100;
      if (qindex != -1)
        qindex = uf_find_peak_root(sInput, sParent, qindex, bucket_value, nullptr);


      cooperative_groups::coalesced_threads().sync();
      sParent[a] = qindex;

      int qy       = qindex / TILE_WIDTH;
      int qx       = qindex % TILE_WIDTH;
      int qgidx    = (by + qy) * pitch + (bx + qx);
      pilot[gidx] = (qindex == -1) ? -1 : qgidx;
    }
    */
  }

  template <int TILE_WIDTH, int TILE_HEIGHT, class T>
  __global__ void add_halo(T* output, int output_width, int output_height, int output_pitch, int ystart, const T* input, int input_pitch)
  {
    int x   = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int y   = ystart + blockIdx.y * TILE_HEIGHT + threadIdx.y;

    if (x >= output_width || y >= output_height)
      return;

    int idx_in =(y - y / TILE_HEIGHT) * input_pitch + (x - x / TILE_WIDTH); 
    int idx_out = y * output_pitch + x;
    output[idx_out] = input[idx_in];
  }

    template <int TILE_WIDTH, int TILE_HEIGHT>
  __global__ void remove_halo(int32_t* output, int output_width, int output_height, int output_pitch, int ystart, const int32_t* input, int input_pitch)
  {
    int x0   = blockIdx.x * blockDim.x + threadIdx.x;
    int y0   = ystart + blockIdx.y * blockDim.y + threadIdx.y;

    if (x0 >= output_width || y0 >= output_height)
      return;

    auto iceil = [](int a, int b) { return (a + b - 1) / b; };

    int y = (y0 == 0) ? 0 : (iceil(y0, TILE_HEIGHT - 1) - 1) * TILE_HEIGHT + (y0 - 1) % (TILE_HEIGHT - 1) + 1;
    int x = (x0 == 0) ? 0 : (iceil(x0, TILE_WIDTH - 1) - 1) * TILE_WIDTH + (x0 - 1) % (TILE_WIDTH - 1) + 1;
    int q  = input[y * input_pitch + x];
    if (q != -1)
    {
      int qy      = q / input_pitch;
      int qx      = q % input_pitch;
      qy = qy - qy / TILE_HEIGHT;
      qx = qx - qx / TILE_WIDTH;   
      assert(qx < output_width && qy < output_height); 
      q = qy * output_pitch + qx;
    }
    output[y0 * output_pitch + x0] = q;
  }

} // namespace

void maxtree_cu(const uint16_t* input_buffer, std::ptrdiff_t input_stride, int width0, int height0, int32_t* parent,
                std::ptrdiff_t parent_stride, uint32_t variant, Chrono* chrono)
{
  const int connectivity = variant & MAXTREE_CONNECTIVITY_MASK;
  assert(connectivity == 4 || connectivity == 8);

  constexpr int TILE_WIDTH  = 64;
  constexpr int TILE_HEIGHT = 16;

  cudaProfilerStart();
  cudaFuncSetCacheConfig(compute_maxtree_tile_HDR<TILE_WIDTH, TILE_HEIGHT>, cudaFuncCachePreferShared);
  // cudaFuncSetCacheConfig(compute_maxtree_tile_optim_1d_connection<TILE_WIDTH, TILE_HEIGHT>,
  // cudaFuncCachePreferShared); cudaFuncSetCacheConfig(compute_maxtree_tile_base<TILE_WIDTH, TILE_HEIGHT>,
  // cudaFuncCachePreferShared); cudaFuncSetCacheConfig(compute_maxtree_tile_optim_1d<TILE_WIDTH, TILE_HEIGHT>,
  // cudaFuncCachePreferShared); cudaFuncSetCacheConfig(merge_maxtree_v, cudaFuncCachePreferL1);
  // cudaFuncSetCacheConfig(merge_maxtree_v_optim, cudaFuncCachePreferL1);
  // cudaFuncSetCacheConfig(merge_maxtree_h, cudaFuncCachePreferL1);
  // cudaFuncSetCacheConfig(merge_maxtree_h_optim, cudaFuncCachePreferL1);
  // cudaFuncSetCacheConfig((void(*)(const uint8_t*, int32_t*, int, int, int, int)) flatten, cudaFuncCachePreferL1);
  int width = width0;
  int height = height0;

  if ((variant & GRID_HALO_ON) != 0)
  {
    width = width0 + (width0 - 1) / (TILE_WIDTH - 1);
    height = height0 + (height0 - 1) / (TILE_HEIGHT - 1);
  }

  using V = uint16_t;
  V*       dInput;
  int32_t* dParent;
  //int32_t* dParentGuide;
  int      pitch; // number of elements between two lines

  cudaError_t err;
  {
    std::size_t dInputPitch, dParentPitch;
    err = cudaMallocPitch(&dInput, &dInputPitch, width * sizeof(V), height);
    checkError(err, "Unable to allocate Input Image memory\n");

    // number of elements between two lines
    pitch = dInputPitch / sizeof(V);
    err   = cudaMallocPitch(&dParent, &dParentPitch, pitch * sizeof(int32_t), height);
    checkError(err, "Unable to allocate Parent memory\n");
    //err = cudaMallocPitch(&dParentGuide, &dParentPitch, pitch * sizeof(int32_t), height);
    //checkError(err, "Unable to allocate Parent memory\n");

    assert(dParentPitch == (pitch * sizeof(int32_t)));
  }


  constexpr int kStreamSplit = 4;
  cudaStream_t  kProcessStream;
  cudaStream_t  kMemoryStream;
  cudaStreamCreate(&kProcessStream);
  cudaStreamCreate(&kMemoryStream);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int count_tile_x      = (width + TILE_WIDTH - 1) / TILE_WIDTH;
  int count_tile_y      = (height + TILE_HEIGHT - 1) / TILE_HEIGHT;
  int kGridRowPerStream = std::max(1, (count_tile_y + kStreamSplit - 1) / kStreamSplit);


  /////////////////////////////////////////////////////////////////////////////////
  //////////////////           LOCAL MAXTREE COMPUTATION on LO_8 ////////////////////
  /////////////////////////////////////////////////////////////////////////////////
  for (int ystart = 0; ystart < height; ystart += TILE_HEIGHT * kGridRowPerStream)
  {
    int nRows     = std::min(height - ystart, kGridRowPerStream * TILE_HEIGHT);
    int nGridRows = (nRows + TILE_HEIGHT - 1) / TILE_HEIGHT;

    // Enqueue copy on stream 1
    if ((variant & GRID_HALO_ON) == 0)
    {
      err = cudaMemcpy2DAsync(dInput + ystart * pitch, pitch * sizeof(V),   //
                              (std::byte*)input_buffer + ystart * input_stride, input_stride, //
                              width * sizeof(V), nRows, cudaMemcpyHostToDevice, kMemoryStream);
      checkError(err, "Unable to copy data with nrows = {}", nRows);
    }
    else
    {
      dim3 threads_per_block(TILE_WIDTH, TILE_HEIGHT);
      dim3 grid_size(count_tile_x, nGridRows);

      // Copy from HOST using with the Unified Memory
      add_halo<TILE_WIDTH, TILE_HEIGHT><<<grid_size, threads_per_block, 0, kMemoryStream>>>(
        dInput, width, height, pitch, ystart, input_buffer, input_stride / sizeof(V));

      err = cudaGetLastError();
      checkError(err, "Unable to execute the kernel ADD HALO");
    }

    // If kernel benchmark, sync and start chrono
    cudaStreamSynchronize(kMemoryStream);

    if (ystart == 0)
      cudaEventRecord(start);

    {
      dim3 threads_per_block(TILE_WIDTH, TILE_HEIGHT);
      dim3 grid_size(count_tile_x, nGridRows);

      compute_maxtree_tile_HDR<TILE_WIDTH, TILE_HEIGHT><<<grid_size, threads_per_block, 0, kProcessStream>>>(
          dInput, dParent, width, height, pitch, ystart, connectivity);

      err = cudaGetLastError();
      checkError(err, "Unable to execute the kernel COMPUTE_MAXTREE_TILE");
    }
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

    // if ((variant & GRID_CONNECTION_OPTIM_MASK) == GRID_CONNECTION_OPTIM_ON)
    //  merge_maxtree_h_optim<<<grid_size, threads_per_block>>>(dInput, dParent, pitch, width, height, TILE_WIDTH,
    //                                                          TILE_HEIGHT, connectivity);
    // else
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

    // if ((variant & GRID_CONNECTION_OPTIM_MASK) == GRID_CONNECTION_OPTIM_ON)
    //  merge_maxtree_v_optim<<<grid_size, threads_per_block>>>(dInput, dParent, pitch, width, height, TILE_HEIGHT,
    //                                                          connectivity);
    // else
    merge_maxtree_v<<<grid_size, threads_per_block>>>(dInput, dParent, pitch, width, height, TILE_HEIGHT, connectivity);
    err = cudaGetLastError();
    checkError(err, "Unable to execute the kernel MERGE");
  }

  /////////////////////////////////////////////////////////////////////////////////
  //////////////////        FLATTENING + COMMIT GLOBAL   ////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////


  {
    const int     count_tile_x      = (width + TILE_WIDTH - 1) / TILE_WIDTH;
    const int     count_tile_y      = (height + TILE_HEIGHT - 1) / TILE_HEIGHT;
    int           kGridRowPerStream = std::max(1, (count_tile_y + kStreamSplit - 1) / kStreamSplit);

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
      if ((variant & GRID_HALO_ON) == 0)
      {
        err = cudaMemcpy2DAsync(parent + ystart * pitch, parent_stride,            //
                                dParent + ystart * pitch, pitch * sizeof(int32_t), //
                                width * sizeof(int32_t), nRows, cudaMemcpyDeviceToHost, kMemoryStream);
        checkError(err, "Unable to copy back data");
      }
      else
      {
        dim3 threads_per_block(TILE_WIDTH, TILE_HEIGHT);
        dim3 grid_size(count_tile_x, nGridRows);

        // Copy from HOST using with the Unified Memory
        remove_halo<TILE_WIDTH, TILE_HEIGHT><<<grid_size, threads_per_block, 0, kMemoryStream>>>(
          parent, width0, height0, parent_stride / sizeof(uint32_t), ystart, dParent, pitch);

        err = cudaGetLastError();
        checkError(err, "Unable to execute the kernel");
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