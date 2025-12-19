#include <mln/core/algorithm/fill.hpp>
#include <mln/core/extension/padding.hpp>
#include <mln/core/image/ndimage.hpp>
#include <mln/core/point.hpp>
#include <mln/core/range/foreach.hpp>
#include <mln/core/algorithm/for_each.hpp>
#include <mln/core/image/view/zip.hpp>
#include <mln/core/colors.hpp>
#include <mln/io/imread.hpp>
#include <mln/io/imsave.hpp>

#include "executor.hpp"
#include "maxtree.hpp"
#include "maxtree_cu.hpp"
#include "tile2d_helpers.hpp"

#include "test_helpers.hpp"
#include <gtest/gtest.h>
#include <mln/core/algorithm/equal.hpp>
#include <mln/io/imprint.hpp>

#include <benchmark/benchmark.h>
#include <fmt/core.h>

Maxtree compute_maxtree(mln::image2d<uint8_t>& f, int connectivity)
{
  Maxtree t;
  t.nodemap = mln::image2d<int>(f.width(), f.height());
  t.parent  = std::make_unique<int[]>(f.width() * f.height());
  t.levels  = std::make_unique<uint8_t[]>(f.width() * f.height());

  { // Prepare nodemap
    auto tmp           = t.nodemap;
    int  borders[2][2] = {{1, 1}, {1, 1}};
    mln::fill(tmp, -1);
    tmp.inflate_domain(1);
    mln::pad(tmp, mln::PAD_CONSTANT, borders, INT32_MAX);
  }

  auto count   = maxtree2d_salembier(f.as_tile(), t.nodemap.as_tile(), t.parent.get(), t.levels.get(), connectivity);
  t.node_count = count;

  return t;
}

void load_images(const char* path, mln::image2d<uint8_t>& input_8, mln::image2d<uint16_t>& input_16, bool& supported_hdr) 
{
  mln::image2d<mln::rgb8> input;
  
  auto ima = mln::io::imread(path);
  if (ima.sample_type() == mln::sample_type_id::RGB8)
  {
    input = *(ima.template cast_to<mln::rgb8, 2>());
    supported_hdr = true;
  }
  else if (ima.sample_type() == mln::sample_type_id::UINT8)
  {
    input_8 = *(ima.template cast_to<uint8_t, 2>());
    return;
  }
  else
    throw("Need gray scale (8-bit) or RGB8 image");

  input_8.resize(input.domain());
  input_16.resize(input.domain());

  mln::for_each(mln::view::zip(input, input_8, input_16), [](auto&& X) {
    auto& [x, y, z] = X;
    float v = 0.2126f * x[0] + 0.7152f * x[1] + 0.0722f * x[2];
    y = v;
    z = v * (UINT16_MAX / UINT8_MAX);
  });
}

struct GBenchmarkChrono : public Chrono
{
  GBenchmarkChrono(benchmark::State* state)
    : m_st{state}
  {
  }

  void ResumeTiming() final { m_st->ResumeTiming(); }
  void PauseTiming() final { m_st->PauseTiming(); }
  void SetIterationTime(double seconds) { m_st->SetIterationTime(seconds); }

private:
  benchmark::State* m_st;
};


class Fixture
{
public:
  static std::string path;
  static bool        no_check;
  static bool        bench_kernel_only;
  static bool        supported_hdr;

  void bench(benchmark::State& st, int connectivity, std::function<Maxtree(mln::image2d<uint8_t>&, int)> callback)
  {
    Maxtree t;

    for (auto _ : st)
      t = callback(m_input, connectivity);

    st.SetBytesProcessed(int64_t(st.iterations()) * int64_t(m_input.width() * m_input.height()));

    if (m_tree)
    {
      auto parent = make_parent_image(t.parent.get(), t.node_count, t.nodemap);
      auto ref    = connectivity == 4 ? m_par_ref : m_par_ref_c8;
      ASSERT_TRUE(compare_debug(m_input, parent, ref));
    }
  }
  void bench_cuda(benchmark::State& st, int32_t variant)
  {
    GBenchmarkChrono         chrono = {&st};
    mln::bp::Tile2D<int32_t> t;
    maxtree_allocate_output(m_input.width(), m_input.height(), sizeof(uint8_t), t);

    cuda::cudaHostRegister(m_input.buffer(), m_input.byte_stride() * m_input.height(), 0);

    for (auto _ : st)
      maxtree_cu(m_input.as_tile(), t, variant, bench_kernel_only ? &chrono : nullptr);

    cuda::cudaHostUnregister(m_input.buffer());

    st.SetBytesProcessed(int64_t(st.iterations()) * int64_t(m_input.width() * m_input.height()));
    if (m_tree)
    {
      int            sizes[2]     = {t.width(), t.height()};
      std::ptrdiff_t strides[2]   = {sizeof(int32_t), t.stride()};
      auto           impar        = mln::image2d<int32_t>::from_buffer(t.data(), sizes, strides);
      auto           parent       = make_parent_image(impar, m_input);
      int            connectivity = variant & MAXTREE_CONNECTIVITY_MASK;
      auto           ref          = connectivity == 4 ? m_par_ref : m_par_ref_c8;
      ASSERT_TRUE(compare_debug(m_input, parent, ref));
    }
  }

  void bench_cuda_hdr(benchmark::State& st, int32_t variant)
  {
    if (Fixture::supported_hdr)
    {
      GBenchmarkChrono         chrono = {&st};
      mln::bp::Tile2D<int32_t> t;
      maxtree_allocate_output(m_input_16.width(), m_input_16.height(), sizeof(uint16_t), t);

      cuda::cudaHostRegister(m_input_16.buffer(), m_input_16.byte_stride() * m_input_16.height(), 0);

      for (auto _ : st)
        maxtree_cu(m_input_16.as_tile(), t, variant, bench_kernel_only ? &chrono : nullptr);

      cuda::cudaHostUnregister(m_input_16.buffer());

      st.SetBytesProcessed(int64_t(st.iterations()) * int64_t(m_input_16.width() * m_input_16.height()));
    }
    else
    {
      st.SkipWithError("Only RGB8 images are supported for HDR.");
    }
  }

  Fixture()
  {
    if (m_input.buffer() == nullptr)
      load_images(path.c_str(), m_input, m_input_16, Fixture::supported_hdr);

    if (m_tree == nullptr && !Fixture::no_check)
    {
      m_tree       = std::make_unique<Maxtree>(compute_maxtree(m_input, 4));
      m_par_ref    = make_parent_image(m_tree->parent.get(), m_tree->node_count, m_tree->nodemap);
      m_tree_c8    = std::make_unique<Maxtree>(compute_maxtree(m_input, 8));
      m_par_ref_c8 = make_parent_image(m_tree_c8->parent.get(), m_tree_c8->node_count, m_tree_c8->nodemap);
    }
  }

  


  void Salembier_ST(benchmark::State& st);
  void Tiled_ST(benchmark::State& st);
  void Tiled_MT(benchmark::State& st);
  void Tiled_1D_MT(benchmark::State& st);
  void CUDA_OPTIM_1D(benchmark::State& st);
  void CUDA_BASE(benchmark::State& st);
  void CUDA_BASE_HDR(benchmark::State& st);
  void CUDA_FL(benchmark::State& st);

private:
  std::unique_ptr<Maxtree>   m_tree    = nullptr;
  std::unique_ptr<Maxtree>   m_tree_c8 = nullptr;
  mln::image2d<uint8_t>      m_input;
  mln::image2d<uint16_t>     m_input_16;
  mln::image2d<mln::point2d> m_par_ref;
  mln::image2d<mln::point2d> m_par_ref_c8;
};

std::string Fixture::path;
bool        Fixture::no_check          = false;
bool        Fixture::bench_kernel_only = false;
bool        Fixture::supported_hdr     = false;



void Fixture::Salembier_ST(benchmark::State& st)
{
  int connectivity = st.range(0);
  this->bench(st, connectivity, compute_maxtree);
}

void Fixture::Tiled_ST(benchmark::State& st)
{
  int  connectivity = st.range(0);
  auto foo          = [](mln::image2d<uint8_t>& input, int connectivity) {
    Executor ex(MaxtreeMethod::Salembier);
    return execute_sequential(&ex, input, connectivity);
  };
  this->bench(st, connectivity, foo);
}

void Fixture::Tiled_MT(benchmark::State& st)
{
  int  connectivity = st.range(0);
  auto foo          = [](mln::image2d<uint8_t>& input, int connectivity) {
    Executor ex(MaxtreeMethod::Salembier);
    return execute_parallel(&ex, input, connectivity);
  };
  this->bench(st, connectivity, foo);
}

void Fixture::Tiled_1D_MT(benchmark::State& st)
{
  int  connectivity = st.range(0);
  auto foo          = [](mln::image2d<uint8_t>& input, int connectivity) {
    Executor ex(MaxtreeMethod::Maxtree1D);
    return execute_parallel(&ex, input, connectivity);
  };
  this->bench(st, connectivity, foo);
}
void Fixture::CUDA_OPTIM_1D(benchmark::State& st)
{
  int connectivity = st.range(0);
  int optim        = st.range(1);
  this->bench_cuda(st, MAXTREE_OPTIM_1D | connectivity | optim);
}

void Fixture::CUDA_BASE(benchmark::State& st)
{
  int connectivity = st.range(0);
  int optim        = st.range(1);
  this->bench_cuda(st, MAXTREE_BASE | connectivity | optim);
}

void Fixture::CUDA_BASE_HDR(benchmark::State& st)
{
  int connectivity = st.range(0);
  int optim        = st.range(1);
  this->bench_cuda_hdr(st, MAXTREE_BASE | connectivity | optim);
}


typedef benchmark::internal::Benchmark benchmark_t;
int main(int argc, char** argv)
{
  using namespace std::placeholders;
  ::benchmark::Initialize(&argc, argv);

  for (int i = 1; i < argc; i++)
    if (argv[i] == std::string_view("--no-check"))
    {
      Fixture::no_check = true;
      std::swap(argv[i], argv[--argc]);
    }
	else if (argv[i] == std::string_view("--bench-kernel-only"))
    {
      Fixture::bench_kernel_only = true;
      std::swap(argv[i], argv[--argc]);
    }

  if (argc < 2)
  {
    fmt::print(stderr, "Expected arguments: {} [--no-check] [google benchmarks params] [images paths...]\n", argv[0]);
    std::exit(1);
  }
  Fixture::path = argv[1];
  fmt::print(stderr, "No checking = {}\n", Fixture::no_check);
  fmt::print(stderr, "Kernel-timing only = {}\n", Fixture::bench_kernel_only);
  
  Fixture fx;
  {
    benchmark_t* b[3];
    b[0] = benchmark::RegisterBenchmark("Salembier_ST", std::bind(&Fixture::Salembier_ST, &fx, _1));
    b[1] = benchmark::RegisterBenchmark("Tiled_ST", std::bind(&Fixture::Tiled_ST, &fx, _1));
    b[2] = benchmark::RegisterBenchmark("Tiled_MT", std::bind(&Fixture::Tiled_MT, &fx, _1));
    b[0]->UseRealTime()->Unit(benchmark::kMillisecond);
    b[1]->UseRealTime()->Unit(benchmark::kMillisecond);
    b[2]->UseRealTime()->Unit(benchmark::kMillisecond);
    b[0]->Arg(4)->Arg(8);
    b[1]->Arg(4)->Arg(8);
    b[2]->Arg(4)->Arg(8);
  }
  
  {
    benchmark_t* b[2];
    b[0] = benchmark::RegisterBenchmark("CUDA_OPTIM_1D", std::bind(&Fixture::CUDA_OPTIM_1D, &fx, _1));
    b[1] = benchmark::RegisterBenchmark("CUDA_BASE", std::bind(&Fixture::CUDA_BASE, &fx, _1));

    b[0]->Args({4, GRID_CONNECTION_OPTIM_OFF})
        ->Args({8, GRID_CONNECTION_OPTIM_OFF})
        ->Args({4, GRID_CONNECTION_OPTIM_ON})
        ->Args({8, GRID_CONNECTION_OPTIM_ON})
        ;
    
    b[1]->Args({4, GRID_CONNECTION_OPTIM_OFF})
        ->Args({8, GRID_CONNECTION_OPTIM_OFF})
        ->Args({4, GRID_CONNECTION_OPTIM_ON})
        ->Args({8, GRID_CONNECTION_OPTIM_ON})
        ;


    if (Fixture::bench_kernel_only)
    {
      b[0]->UseManualTime()->Unit(benchmark::kMillisecond);
      b[1]->UseManualTime()->Unit(benchmark::kMillisecond);
    }
    else
    {
      b[0]->UseRealTime()->Unit(benchmark::kMillisecond);
      b[1]->UseRealTime()->Unit(benchmark::kMillisecond);
    }
  }

  {
    benchmark_t* b[2];
    b[0] = benchmark::RegisterBenchmark("CUDA_BASE_HDR", std::bind(&Fixture::CUDA_BASE_HDR, &fx, _1));
    b[0]->Args({4, GRID_CONNECTION_OPTIM_OFF});
    b[1] = benchmark::RegisterBenchmark("CUDA_HALO_HDR", std::bind(&Fixture::CUDA_BASE_HDR, &fx, _1));
    b[1]->Args({4, GRID_HALO_ON});


    if (Fixture::bench_kernel_only)
    {
      b[0]->UseManualTime()->Unit(benchmark::kMillisecond);
      b[1]->UseManualTime()->Unit(benchmark::kMillisecond);
    }
    else
    {
      b[0]->UseRealTime()->Unit(benchmark::kMillisecond);
      b[1]->UseRealTime()->Unit(benchmark::kMillisecond);
    }
  }
  
  ::benchmark::RunSpecifiedBenchmarks();
}
