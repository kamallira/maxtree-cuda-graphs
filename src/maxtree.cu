
// maxtree.cu
#include <fmt/core.h>
#include "maxtree.cuh"           // kernels, masks/constants, Chrono*, ENABLE_KERNEL_TIMERS, 'timers' symbol
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cassert>
#include <algorithm>
#include <vector>
#include <cstdint>
#include <cstddef>

namespace {
    template <class... TArgs>
    inline void checkError(cudaError_t err, const char* msg, TArgs&&... args)
    {
        if (err != cudaSuccess) {
            fmt::print(stderr, msg, std::forward<TArgs>(args)...);
            fmt::print(stderr, "\nCUDA msg: {}\n", cudaGetErrorString(err));
            std::abort();
        }
    }
} // anonymous

// -----------------------------------------------------------------------------
// Wrappers expected by main.cpp (namespace cuda::*)
// -----------------------------------------------------------------------------
namespace cuda {
    int cudaHostRegister(void* ptr, size_t size, unsigned int flags)
    {
        auto err = ::cudaHostRegister(ptr, size, flags);
        if (err != cudaSuccess) {
            fmt::print(stderr, "Unable to pin memory\nCUDA msg: {}\n", cudaGetErrorString(err));
        }
        return err;
    }

    int cudaHostUnregister(void* ptr)
    {
        return ::cudaHostUnregister(ptr);
    }

    void cudaFreeHost(void* ptr)
    {
        auto err = ::cudaFreeHost(ptr);
        if (err != cudaSuccess) {
            fmt::print(stderr, "Unable to release memory\nCUDA msg: {}\n", cudaGetErrorString(err));
        }
    }
} // namespace cuda

// -----------------------------------------------------------------------------
// Allocation helper with exact signature the linker expects
// -----------------------------------------------------------------------------
void maxtree_allocate_output(int** ptr, int width, int height, int size, int* pitch)
{
    constexpr uint32_t align = 512;
    int linebytes = (width * size + align - 1) & ~(align - 1);
    int linesize  = linebytes / size;

    auto err = ::cudaMallocHost(reinterpret_cast<void**>(ptr),
                                static_cast<size_t>(linesize) * sizeof(int) * static_cast<size_t>(height));
    if (err != cudaSuccess) {
        fmt::print(stderr, "Unable to allocate HOST output\nCUDA msg: {}\n", cudaGetErrorString(err));
        std::abort();
    }
    *pitch = linesize;
}

// -----------------------------------------------------------------------------
// Persistent context with one-time graph capture/instantiate per configuration
// -----------------------------------------------------------------------------
namespace {
    struct Ctx {
        // Current configuration
        int      width   = 0;
        int      height  = 0;
        uint32_t variant = 0;

        // Host pointers/strides (bytes per row)
        const uint8_t*   hInput      = nullptr;
        std::ptrdiff_t   hInputPitch = 0;  // bytes/row
        int32_t*         hOutput     = nullptr;
        std::ptrdiff_t   hOutputPitch= 0;  // bytes/row

        // Device allocations
        uint8_t*  dInput   = nullptr;
        int32_t*  dParent  = nullptr;
        size_t    dInputPitchB  = 0;       // bytes/row
        size_t    dParentPitchB = 0;       // bytes/row

        // Streams/Events
        cudaStream_t processStream = nullptr;
        cudaStream_t memoryStream  = nullptr;
        cudaEvent_t  start         = nullptr;
        cudaEvent_t  stop          = nullptr;

        // Graph-only per-band events
        std::vector<cudaEvent_t> evtJoinAtBegin;   // size 1 (process->memory)
        std::vector<cudaEvent_t> evtMemToProc;     // H->D bands
        std::vector<cudaEvent_t> evtProcToMem;     // flatten bands
        std::vector<cudaEvent_t> evtJoinAtEnd;     // size 1 (memory->process)

        // Graph
        cudaGraph_t     graph     = nullptr;
        cudaGraphExec_t graphExec = nullptr;

        // Flags
        bool resources_ready   = false;    // streams/device buffers created for current config
        bool initialized_graph = false;    // graph captured & instantiated for current config

        static int elements_pitch(size_t pitch_bytes, size_t elem_size) {
            return static_cast<int>(pitch_bytes / elem_size);
        }

        void destroy_events(std::vector<cudaEvent_t>& v) {
            for (auto& e : v) if (e) { cudaEventDestroy(e); e = nullptr; }
            v.clear();
        }

        // Create or reuse resources for (input, output, width, height, variant)
        void init_common(const uint8_t* input_buffer,
                         std::ptrdiff_t input_stride,
                         int w, int h,
                         int32_t* parent, std::ptrdiff_t parent_stride,
                         uint32_t var)
        {
            const bool same =
                (resources_ready &&
                 w == width && h == height && var == variant &&
                 input_buffer == hInput && input_stride == hInputPitch &&
                 parent == hOutput && parent_stride == hOutputPitch);

            // Reuse existing resources when configuration is unchanged
            if (same) {
                return;
            }

            // If configuration changed, tear down old resources
            if (resources_ready) {
                shutdown();
            }

            // Record new configuration
            width   = w;
            height  = h;
            variant = var;
            hInput       = input_buffer;
            hInputPitch  = input_stride;
            hOutput      = parent;
            hOutputPitch = parent_stride;

            // Sanity: host pitches
            {
                const size_t inWidthB  = static_cast<size_t>(width) * sizeof(uint8_t);
                const size_t outWidthB = static_cast<size_t>(width) * sizeof(int32_t);
                if (!(inWidthB  <= static_cast<size_t>(hInputPitch) &&
                      outWidthB <= static_cast<size_t>(hOutputPitch))) {
                    fmt::print(stderr,
                        "[SANITY] Host pitch smaller than row width: inWidthB={} hInputPitch={} outWidthB={} hOutputPitch={}\n",
                        inWidthB, static_cast<size_t>(hInputPitch), outWidthB, static_cast<size_t>(hOutputPitch));
                    std::abort();
                }
            }

            // Streams & events
            checkError(cudaStreamCreate(&processStream), "Unable to create processStream");
            checkError(cudaStreamCreate(&memoryStream),  "Unable to create memoryStream");
            checkError(cudaEventCreate(&start),          "Unable to create start event");
            checkError(cudaEventCreate(&stop),           "Unable to create stop event");

            // Device allocations
            size_t inPitchB = 0, parPitchB = 0;
            checkError(cudaMallocPitch(&dInput,  &inPitchB,  width * sizeof(uint8_t), height),
                       "Unable to allocate Input Image memory\n");
            checkError(cudaMallocPitch(&dParent, &parPitchB, inPitchB * sizeof(int32_t), height),
                       "Unable to allocate Parent memory\n");
            dInputPitchB  = inPitchB;
            dParentPitchB = parPitchB;

            // Sanity: device pitches
            {
                const size_t inWidthB  = static_cast<size_t>(width) * sizeof(uint8_t);
                const size_t outWidthB = static_cast<size_t>(width) * sizeof(int32_t);
                if (!(inWidthB  <= dInputPitchB && outWidthB <= dParentPitchB)) {
                    fmt::print(stderr,
                        "[SANITY] Device pitch smaller than row width: inWidthB={} dInputPitchB={} outWidthB={} dParentPitchB={}\n",
                        inWidthB, dInputPitchB, outWidthB, dParentPitchB);
                    std::abort();
                }
            }

            // Kernel cache config (as in your original)
            constexpr int TW = 64, TH = 16;
            checkError(cudaFuncSetCacheConfig(compute_maxtree_tile_base_optim<TW, TH>, cudaFuncCachePreferShared), "");
            checkError(cudaFuncSetCacheConfig(compute_maxtree_tile_optim_1d_connection<TW, TH>, cudaFuncCachePreferShared), "");
            checkError(cudaFuncSetCacheConfig(compute_maxtree_tile_base<TW, TH>, cudaFuncCachePreferShared), "");
            checkError(cudaFuncSetCacheConfig(compute_maxtree_tile_optim_1d<TW, TH>, cudaFuncCachePreferShared), "");
            checkError(cudaFuncSetCacheConfig((void(*)(const uint8_t*, int32_t*, int, int, int, int)) flatten, cudaFuncCachePreferL1), "");
            checkError(cudaFuncSetCacheConfig(merge_maxtree_v<uint8_t>,  cudaFuncCachePreferL1), "");
            checkError(cudaFuncSetCacheConfig(merge_maxtree_v_optim,     cudaFuncCachePreferL1), "");
            checkError(cudaFuncSetCacheConfig(merge_maxtree_h<uint8_t>,  cudaFuncCachePreferL1), "");
            checkError(cudaFuncSetCacheConfig(merge_maxtree_h_optim,     cudaFuncCachePreferL1), "");

            // Reset device timers once (if enabled)
            unsigned long long _timers[5] = {0};
            if constexpr (ENABLE_KERNEL_TIMERS) {
                checkError(cudaMemcpyToSymbolAsync(timers, _timers, sizeof(_timers), 0,
                                                   cudaMemcpyHostToDevice, processStream),
                           "Unable to initialize kernel timers\n");
            }

            // Mark resources ready; graph will be set by subsequent steps
            resources_ready   = true;
            initialized_graph = false;
        }

        // --------- Graph capture & instantiate (once per configuration) ----------
        void init_graph()
        {
            // Pre-compute bands & create per-band events
            constexpr int TILE_WIDTH  = 64;
            constexpr int TILE_HEIGHT = 16;
            constexpr int kStreamSplit = 4;

            const int count_tile_y = (height + TILE_HEIGHT - 1) / TILE_HEIGHT;
            const int kGridRowPerStream = std::max(1, (count_tile_y + kStreamSplit - 1) / kStreamSplit);
            const int bandH2D = (height + (TILE_HEIGHT * kGridRowPerStream) - 1) / (TILE_HEIGHT * kGridRowPerStream);

            constexpr int FH = 32;
            const int count_y = (height + FH - 1) / FH;
            const int kGridRowPerStreamF = std::max(1, (count_y + kStreamSplit - 1) / kStreamSplit);
            const int bandDtoH = (height + (FH * kGridRowPerStreamF) - 1) / (FH * kGridRowPerStreamF);

            evtJoinAtBegin.resize(1);
            for (auto& e : evtJoinAtBegin) checkError(cudaEventCreate(&e), "evtJoinAtBegin create failed");
            evtMemToProc.resize(bandH2D);
            for (auto& e : evtMemToProc)   checkError(cudaEventCreate(&e), "evtMemToProc create failed");
            evtProcToMem.resize(bandDtoH);
            for (auto& e : evtProcToMem)   checkError(cudaEventCreate(&e), "evtProcToMem create failed");
            evtJoinAtEnd.resize(1);
            for (auto& e : evtJoinAtEnd)   checkError(cudaEventCreate(&e), "evtJoinAtEnd create failed");

            // Capture (thread-local)
            checkError(cudaStreamBeginCapture(processStream, cudaStreamCaptureModeThreadLocal),
                       "Capture Begin failed");

            // Join memory stream into capture
            checkError(cudaEventRecord(evtJoinAtBegin[0], processStream), "begin event record failed");
            checkError(cudaStreamWaitEvent(memoryStream, evtJoinAtBegin[0], 0), "begin stream wait failed");

            // Start timing
            checkError(cudaEventRecord(start, processStream), "start record failed");

            const int connectivity = variant & MAXTREE_CONNECTIVITY_MASK;
            assert(connectivity == 4 || connectivity == 8);

            // HtoD + compute bands
            {
                int bandIdx = 0;
                const int count_tile_x = (width + TILE_WIDTH - 1) / TILE_WIDTH;

                for (int ystart = 0; ystart < height; ystart += TILE_HEIGHT * kGridRowPerStream, ++bandIdx) {
                    const int nRows     = std::min(height - ystart, kGridRowPerStream * TILE_HEIGHT);
                    const int nGridRows = (nRows + TILE_HEIGHT - 1) / TILE_HEIGHT;

                    // HtoD (memory stream)
                    checkError(cudaMemcpy2DAsync(
                        dInput + ystart * elements_pitch(dInputPitchB, sizeof(uint8_t)), dInputPitchB,
                        hInput + ystart * hInputPitch,                                   hInputPitch,
                        width * sizeof(uint8_t), nRows, cudaMemcpyHostToDevice, memoryStream),
                        "HtoD memcpy2DAsync failed");

                    // Mem->Proc dependency (unique event per band)
                    checkError(cudaEventRecord(evtMemToProc[bandIdx], memoryStream), "mem->proc event record failed");
                    checkError(cudaStreamWaitEvent(processStream, evtMemToProc[bandIdx], 0), "mem->proc wait failed");

                    // Compute (process stream)
                    if ((variant & MAXTREE_ALGORITH_MASK) == MAXTREE_OPTIM_1D) {
                        dim3 threads(TILE_WIDTH, 1);
                        dim3 grid(count_tile_x, nGridRows);
                        if ((variant & GRID_CONNECTION_OPTIM_MASK) == GRID_CONNECTION_OPTIM_ON) {
                            compute_maxtree_tile_optim_1d_connection<TILE_WIDTH, TILE_HEIGHT><<<grid, threads, 0, processStream>>>(
                                dInput, dParent, width, height,
                                elements_pitch(dInputPitchB, sizeof(uint8_t)), ystart, connectivity);
                        } else {
                            compute_maxtree_tile_optim_1d<TILE_WIDTH, TILE_HEIGHT><<<grid, threads, 0, processStream>>>(
                                dInput, dParent, width, height,
                                elements_pitch(dInputPitchB, sizeof(uint8_t)), ystart, connectivity);
                        }
                    } else { // MAXTREE_BASE
                        dim3 threads(TILE_WIDTH, TILE_HEIGHT);
                        dim3 grid(count_tile_x, nGridRows);
                        if ((variant & GRID_CONNECTION_OPTIM_MASK) == GRID_CONNECTION_OPTIM_ON) {
                            compute_maxtree_tile_base_optim<TILE_WIDTH, TILE_HEIGHT><<<grid, threads, 0, processStream>>>(
                                dInput, dParent, width, height,
                                elements_pitch(dInputPitchB, sizeof(uint8_t)), ystart, connectivity);
                        } else {
                            compute_maxtree_tile_base<TILE_WIDTH, TILE_HEIGHT><<<grid, threads, 0, processStream>>>(
                                dInput, dParent, width, height,
                                elements_pitch(dInputPitchB, sizeof(uint8_t)), ystart, connectivity);
                        }
                    }
                }
            }

            // Merge H
            {
                constexpr int TILE_WIDTH = 64;
                const int count_tile_x = (width + TILE_WIDTH - 1) / TILE_WIDTH;
                if (count_tile_x > 1) {
                    constexpr int NX = 32, NY = 32;
                    dim3 threads(NX, NY);
                    dim3 grid((count_tile_x - 1 + NX - 1)/NX, (height + NY - 1)/NY);
                    if ((variant & GRID_CONNECTION_OPTIM_MASK) == GRID_CONNECTION_OPTIM_ON) {
                        merge_maxtree_h_optim<<<grid, threads, 0, processStream>>>(
                            dInput, dParent, elements_pitch(dInputPitchB, sizeof(uint8_t)),
                            width, height, TILE_WIDTH, 16, (variant & MAXTREE_CONNECTIVITY_MASK));
                    } else {
                        merge_maxtree_h<uint8_t><<<grid, threads, 0, processStream>>>(
                            dInput, dParent, elements_pitch(dInputPitchB, sizeof(uint8_t)),
                            width, height, TILE_WIDTH, (variant & MAXTREE_CONNECTIVITY_MASK));
                    }
                }
            }

            // Merge V
            {
                constexpr int TILE_HEIGHT = 16;
                const int count_tile_y = (height + TILE_HEIGHT - 1) / TILE_HEIGHT;
                if (count_tile_y > 1) {
                    constexpr int NX = 256, NY = 4;
                    dim3 threads(NX, NY);
                    dim3 grid((width + NX - 1)/NX, (count_tile_y - 1 + NY - 1)/NY);
                    if ((variant & GRID_CONNECTION_OPTIM_MASK) == GRID_CONNECTION_OPTIM_ON) {
                        merge_maxtree_v_optim<<<grid, threads, 0, processStream>>>(
                            dInput, dParent, elements_pitch(dInputPitchB, sizeof(uint8_t)),
                            width, height, TILE_HEIGHT, (variant & MAXTREE_CONNECTIVITY_MASK));
                    } else {
                        merge_maxtree_v<uint8_t><<<grid, threads, 0, processStream>>>(
                            dInput, dParent, elements_pitch(dInputPitchB, sizeof(uint8_t)),
                            width, height, TILE_HEIGHT, (variant & MAXTREE_CONNECTIVITY_MASK));
                    }
                }
            }

            // Flatten + DtoH
            {
                constexpr int FW = 32, FH = 32;
                const int count_x = (width  + FW - 1) / FW;
                const int count_y = (height + FH - 1) / FH;
                constexpr int kStreamSplitF = 4;
                const int kGridRowPerStreamF = std::max(1, (count_y + kStreamSplitF - 1)/kStreamSplitF);

                int bandIdx = 0;
                for (int ystart = 0; ystart < height; ystart += kGridRowPerStreamF * FH, ++bandIdx) {
                    const int nRows     = std::min(height - ystart, kGridRowPerStreamF * FH);
                    const int nGridRows = (nRows + FH - 1) / FH;
                    dim3 threads(FW, FH);
                    dim3 grid(count_x, nGridRows);

                    flatten<<<grid, threads, 0, processStream>>>(
                        dInput, dParent, width, height,
                        elements_pitch(dInputPitchB, sizeof(uint8_t)), ystart);

                    // Proc->Mem dependency (unique event per band)
                    checkError(cudaEventRecord(evtProcToMem[bandIdx], processStream), "proc->mem event record failed");
                    checkError(cudaStreamWaitEvent(memoryStream, evtProcToMem[bandIdx], 0), "proc->mem wait failed");

                    // DtoH (memory stream)
                    checkError(cudaMemcpy2DAsync(
                        reinterpret_cast<uint8_t*>(hOutput) + ystart * hOutputPitch, hOutputPitch,
                        dParent + ystart * elements_pitch(dParentPitchB, sizeof(int32_t)), dParentPitchB,
                        width * sizeof(int32_t), nRows, cudaMemcpyDeviceToHost, memoryStream),
                        "DtoH memcpy2DAsync failed");
                }
            }

            // Re-join memory -> process
            checkError(cudaEventRecord(evtJoinAtEnd[0], memoryStream), "end event record failed");
            checkError(cudaStreamWaitEvent(processStream, evtJoinAtEnd[0], 0), "end stream wait failed");

            // Stop timing
            checkError(cudaEventRecord(stop, processStream), "stop record failed");

            // End capture, instantiate once, upload
            checkError(cudaStreamEndCapture(processStream, &graph), "Capture End failed");
            checkError(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0), "Instantiate failed");
            checkError(cudaGraphUpload(graphExec, processStream), "Graph Upload failed");

            initialized_graph = true;
        }

        // --------- Graph launch for subsequent iterations ----------
        void run_graph(Chrono* chrono)
        {
            auto gerr = cudaGraphLaunch(graphExec, processStream);
            if (gerr != cudaSuccess) {
                fmt::print(stderr, "[GRAPH-LAUNCH] error: {}\n", cudaGetErrorString(gerr));
                checkError(gerr, "Launch failed");
            }
            auto serr = cudaStreamSynchronize(processStream);
            if (serr != cudaSuccess) {
                fmt::print(stderr,
                    "[STREAM-SYNC] failed with {}\n  width={} height={} dInputPitchB={} dParentPitchB={} hInputPitch={} hOutputPitch={}\n",
                    cudaGetErrorString(serr),
                    width, height, dInputPitchB, dParentPitchB, static_cast<size_t>(hInputPitch), static_cast<size_t>(hOutputPitch));
                checkError(serr, "processStream sync failed");
            }

            if (chrono) {
                float ms = 0.f;
                checkError(cudaEventElapsedTime(&ms, start, stop), "EventElapsedTime failed");
                chrono->SetIterationTime(ms / 1000.0);
            }
        }

        void shutdown()
        {
            if (graphExec) { cudaGraphExecDestroy(graphExec); graphExec = nullptr; }
            if (graph)     { cudaGraphDestroy(graph);         graph     = nullptr; }

            if (dInput)    { cudaFree(dInput);   dInput   = nullptr; }
            if (dParent)   { cudaFree(dParent);  dParent  = nullptr; }

            if (processStream) { cudaStreamDestroy(processStream); processStream = nullptr; }
            if (memoryStream)  { cudaStreamDestroy(memoryStream);  memoryStream  = nullptr; }

            if (start) { cudaEventDestroy(start); start = nullptr; }
            if (stop)  { cudaEventDestroy(stop);  stop  = nullptr; }

            destroy_events(evtJoinAtBegin);
            destroy_events(evtMemToProc);
            destroy_events(evtProcToMem);
            destroy_events(evtJoinAtEnd);

            resources_ready   = false;
            initialized_graph = false;
        }
    };

    // Singleton context + atexit cleanup
    static Ctx* g_ctx = nullptr;
    static Ctx& ctx() { if (!g_ctx) g_ctx = new Ctx(); return *g_ctx; }
    static void _atexit_cleanup() { if (g_ctx) { g_ctx->shutdown(); delete g_ctx; g_ctx = nullptr; } }
} // anonymous

// -----------------------------------------------------------------------------
// Public API (unchanged signature expected by your existing main.cpp)
// -----------------------------------------------------------------------------
void maxtree_cu(const uint8_t* input_buffer, std::ptrdiff_t input_stride,
                int width, int height,
                int32_t* parent, std::ptrdiff_t parent_stride,
                uint32_t variant, Chrono* chrono)
{
    // Ensure cleanup at process exit
    static bool atexit_registered = false;
    if (!atexit_registered) { std::atexit(_atexit_cleanup); atexit_registered = true; }

    // Create or reuse resources for this configuration
    ctx().init_common(input_buffer, input_stride, width, height, parent, parent_stride, variant);

    // Capture & instantiate graph ONCE per configuration
    if (!ctx().initialized_graph) {
        ctx().init_graph();
        ctx().initialized_graph = true;   // subsequent calls launch only
    }

    // Launch graph for this iteration
    ctx().run_graph(chrono);
}
