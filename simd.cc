#include <benchmark/benchmark.h>
#include <immintrin.h>

#include <iostream>
#include <memory>
#include <string>

namespace {

constexpr size_t align_size = 16;
const size_t dim = 1024;
int N;

float *query, *data;

float L2Sqr(const float* x, const float* y, const size_t d) {
  float result = 0.0;
  for (int i = 0; i < d; ++i) {
    result += (x[i] - y[i]) * (x[i] - y[i]);
  }
  return result;
}

}  // namespace

namespace fast {

float L2SqrAVX(const float* x, const float* y, const size_t d) {
  __m256 sum1 = _mm256_setzero_ps(), sum2 = _mm256_setzero_ps(), xx1, yy1, xx2,
         yy2, t1, t2;
  const float* end = x + d;
  while (x < end) {
    xx1 = _mm256_loadu_ps(x);
    x += 8;
    yy1 = _mm256_loadu_ps(y);
    y += 8;
    t1 = _mm256_sub_ps(xx1, yy1);
    xx2 = _mm256_loadu_ps(x);
    x += 8;
    yy2 = _mm256_loadu_ps(y);
    y += 8;
    t2 = _mm256_sub_ps(xx2, yy2);
    sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(t1, t1));
    sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(t2, t2));
  }
  sum1 = _mm256_add_ps(sum1, sum2);
  __m128 sumh =
      _mm_add_ps(_mm256_castps256_ps128(sum1), _mm256_extractf128_ps(sum1, 1));
  __m128 tmp1 = _mm_add_ps(sumh, _mm_movehl_ps(sumh, sumh));
  __m128 tmp2 = _mm_add_ps(tmp1, _mm_movehdup_ps(tmp1));
  return _mm_cvtss_f32(tmp2);
}

}  // namespace fast

namespace faiss {

// reads 0 <= d < 4 floats as __m128
static inline __m128 masked_read(int d, const float* x) {
  assert(0 <= d && d < 4);
  __attribute__((__aligned__(16))) float buf[4] = {0, 0, 0, 0};
  switch (d) {
    case 3:
      buf[2] = x[2];
    case 2:
      buf[1] = x[1];
    case 1:
      buf[0] = x[0];
  }
  return _mm_load_ps(buf);
  // cannot use AVX2 _mm_mask_set1_epi32
}

// reads 0 <= d < 8 floats as __m256
static inline __m256 masked_read_8(int d, const float* x) {
  assert(0 <= d && d < 8);
  if (d < 4) {
    __m256 res = _mm256_setzero_ps();
    res = _mm256_insertf128_ps(res, masked_read(d, x), 0);
    return res;
  } else {
    __m256 res = _mm256_setzero_ps();
    res = _mm256_insertf128_ps(res, _mm_loadu_ps(x), 0);
    res = _mm256_insertf128_ps(res, masked_read(d - 4, x + 4), 1);
    return res;
  }
}

float fvec_L2sqr_avx(const float* x, const float* y, size_t d) {
  __m256 msum1 = _mm256_setzero_ps();

  while (d >= 8) {
    __m256 mx = _mm256_loadu_ps(x);
    x += 8;
    __m256 my = _mm256_loadu_ps(y);
    y += 8;
    const __m256 a_m_b1 = mx - my;
    msum1 += a_m_b1 * a_m_b1;
    d -= 8;
  }

  __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
  msum2 += _mm256_extractf128_ps(msum1, 0);

  if (d >= 4) {
    __m128 mx = _mm_loadu_ps(x);
    x += 4;
    __m128 my = _mm_loadu_ps(y);
    y += 4;
    const __m128 a_m_b1 = mx - my;
    msum2 += a_m_b1 * a_m_b1;
    d -= 4;
  }

  if (d > 0) {
    __m128 mx = masked_read(d, x);
    __m128 my = masked_read(d, y);
    __m128 a_m_b1 = mx - my;
    msum2 += a_m_b1 * a_m_b1;
  }

  msum2 = _mm_hadd_ps(msum2, msum2);
  msum2 = _mm_hadd_ps(msum2, msum2);
  return _mm_cvtss_f32(msum2);
}

}  // namespace faiss

namespace hnswlib {

float L2SqrSIMD16ExtAVX(const void* pVect1v, const void* pVect2v,
                        const void* qty_ptr) {
  float* pVect1 = (float*)pVect1v;
  float* pVect2 = (float*)pVect2v;
  size_t qty = *((size_t*)qty_ptr);
  float __attribute__((aligned(32))) TmpRes[8];
  size_t qty16 = qty >> 4;

  const float* pEnd1 = pVect1 + (qty16 << 4);

  __m256 diff, v1, v2;
  __m256 sum = _mm256_set1_ps(0);

  while (pVect1 < pEnd1) {
    v1 = _mm256_loadu_ps(pVect1);
    pVect1 += 8;
    v2 = _mm256_loadu_ps(pVect2);
    pVect2 += 8;
    diff = _mm256_sub_ps(v1, v2);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

    v1 = _mm256_loadu_ps(pVect1);
    pVect1 += 8;
    v2 = _mm256_loadu_ps(pVect2);
    pVect2 += 8;
    diff = _mm256_sub_ps(v1, v2);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
  }

  _mm256_store_ps(TmpRes, sum);
  return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] +
         TmpRes[6] + TmpRes[7];
}

}  // namespace hnswlib

static void Naive(benchmark::State& state) {
  // Code inside this loop is measured repeatedly
  for (int i = 0; auto _ : state) {
    ++i %= N;
    float dist = L2Sqr(query, data + i * dim, dim);
    benchmark::DoNotOptimize(dist);
  }
}

static void FastL2Sqr(benchmark::State& state) {
  // Code inside this loop is measured repeatedly
  for (int i = 0; auto _ : state) {
    ++i %= N;
    float dist = fast::L2SqrAVX(query, data + i * dim, dim);
    benchmark::DoNotOptimize(dist);
  }
}

static void FaissL2Sqr(benchmark::State& state) {
  // Code inside this loop is measured repeatedly
  for (int i = 0; auto _ : state) {
    ++i %= N;
    float dist = faiss::fvec_L2sqr_avx(query, data + i * dim, dim);
    benchmark::DoNotOptimize(dist);
  }
}

static void HNSWL2Sqr(benchmark::State& state) {
  // Code inside this loop is measured repeatedly
  for (int i = 0; auto _ : state) {
    ++i %= N;
    float dist = hnswlib::L2SqrSIMD16ExtAVX(query, data + i * dim, &dim);
    benchmark::DoNotOptimize(dist);
  }
}

int main(int argc, char** argv) {
  N = std::stoll(argv[1]);
  query = (float*)aligned_alloc(64, dim * sizeof(float));
  data = (float*)aligned_alloc(64, dim * N * sizeof(float));
  for (int i = 0; i < dim; ++i) query[i] = rand() % 100;
  for (int i = 0; i < N * dim; ++i) data[i] = rand() % 100;
  // BENCHMARK(Naive);
  BENCHMARK(FastL2Sqr);
  BENCHMARK(FaissL2Sqr);
  BENCHMARK(HNSWL2Sqr);
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  free(query);
  free(data);
}