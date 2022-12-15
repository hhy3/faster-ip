#include <benchmark/benchmark.h>
#include <immintrin.h>

#include <iostream>
#include <memory>
#include <numeric>
#include <string>

int N;
int dim;

float *query, *data;
int *perm;

void global_init() {
  query = (float *)aligned_alloc(64, dim * sizeof(float));
  data = (float *)aligned_alloc(64, dim * N * sizeof(float));
  for (int i = 0; i < dim; ++i) {
    query[i] = rand() / double(RAND_MAX);
  }
  for (int i = 0; i < N * dim; ++i) {
    data[i] = rand() / double(RAND_MAX);
  }
  perm = new int[N];
  std::iota(perm, perm + N, 0);
  std::random_shuffle(perm, perm + N);
}

void global_destroy() {
  free(query);
  free(data);
  delete[] perm;
}

namespace naive {

float IP(const int idx, const size_t d) {
  const float *x = query, *y = data + idx * d;
  float result = 0.0;
  for (int i = 0; i < d; ++i) {
    result += x[i] * y[i];
  }
  return result;
}

}  // namespace naive

namespace checker {

std::vector<double> gt;

void init() {
  gt.resize(N);
  for (int i = 0; i < N; ++i) {
    double d = naive::IP(i, dim);
    gt[i] = d;
  }
}

double check(const auto &func) {
  std::vector<double> cur(N);
  double error = 0.0;
  for (int i = 0; i < N; ++i) {
    double d = func(i, dim);
    error += std::abs(d - gt[i]) / gt[i];
  }
  error /= N;
  std::cout << error << std::endl;
  return error;
}

}  // namespace checker

namespace faiss {

// reads 0 <= d < 4 floats as __m128
static inline __m128 masked_read(int d, const float *x) {
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
static inline __m256 masked_read_8(int d, const float *x) {
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

float IP(const int idx, size_t d) {
  const float *x = query, *y = data + idx * d;
  __m256 msum1 = _mm256_setzero_ps();
  while (d >= 8) {
    __m256 mx = _mm256_loadu_ps(x);
    x += 8;
    __m256 my = _mm256_loadu_ps(y);
    y += 8;
    msum1 += mx * my;
    d -= 8;
  }
  __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
  msum2 += _mm256_extractf128_ps(msum1, 0);
  msum2 = _mm_hadd_ps(msum2, msum2);
  msum2 = _mm_hadd_ps(msum2, msum2);
  return _mm_cvtss_f32(msum2);
}

}  // namespace faiss

namespace fast {

float IP(const int idx, const size_t d) {
  const float *x = query, *y = data + idx * d;
  __m256 sum1 = _mm256_setzero_ps(), sum2 = _mm256_setzero_ps();
  const float *end = x + d;
  while (x < end) {
    {
      auto xx = _mm256_loadu_ps(x);
      x += 8;
      auto yy = _mm256_loadu_ps(y);
      y += 8;
      sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(xx, yy));
    }
    {
      auto xx = _mm256_loadu_ps(x);
      x += 8;
      auto yy = _mm256_loadu_ps(y);
      y += 8;
      sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(xx, yy));
    }
  }
  sum1 = _mm256_add_ps(sum1, sum2);
  auto sumh =
      _mm_add_ps(_mm256_castps256_ps128(sum1), _mm256_extractf128_ps(sum1, 1));
  auto tmp1 = _mm_add_ps(sumh, _mm_movehl_ps(sumh, sumh));
  auto tmp2 = _mm_add_ps(tmp1, _mm_movehdup_ps(tmp1));
  return _mm_cvtss_f32(tmp2);
}

}  // namespace fast

namespace fast_avx512 {

float IP(const int idx, const size_t d) {
  const float *x = query, *y = data + idx * d;
  __m512 sum1 = _mm512_setzero_ps(), sum2 = _mm512_setzero_ps();
  const float *end = x + d;
  while (x < end) {
    {
      auto xx = _mm512_loadu_ps(x);
      x += 16;
      auto yy = _mm512_loadu_ps(y);
      y += 16;
      sum1 = _mm512_add_ps(sum1, _mm512_mul_ps(xx, yy));
    }
    {
      auto xx = _mm512_loadu_ps(x);
      x += 16;
      auto yy = _mm512_loadu_ps(y);
      y += 16;
      sum2 = _mm512_add_ps(sum2, _mm512_mul_ps(xx, yy));
    }
  }
  sum1 = _mm512_add_ps(sum1, sum2);
  auto sumhh = _mm256_add_ps(_mm512_castps512_ps256(sum1),
                             _mm512_extractf32x8_ps(sum1, 1));
  auto sumh = _mm_add_ps(_mm256_castps256_ps128(sumhh),
                         _mm256_extractf128_ps(sumhh, 1));
  auto tmp1 = _mm_add_ps(sumh, _mm_movehl_ps(sumh, sumh));
  auto tmp2 = _mm_add_ps(tmp1, _mm_movehdup_ps(tmp1));
  return _mm_cvtss_f32(tmp2);
}

}  // namespace fast_avx512

namespace bf16 {

uint16_t *code;

void FloatToBFloat16(const float *src, void *dst, int64_t size) {
  const uint16_t *p = reinterpret_cast<const uint16_t *>(src);
  uint16_t *q = reinterpret_cast<uint16_t *>(dst);
  for (; size != 0; p += 2, q++, size--) {
    *q = p[1];
  }
}

void init() {
  code = (uint16_t *)aligned_alloc(64, N * dim * 2);
  for (int i = 0; i < N; ++i) {
    FloatToBFloat16(data + i * dim, code + i * dim, dim);
  }
}

float IP(const int idx, const int d) {
  const float *x = query;
  const uint16_t *y = code + idx * d;
  __m256 sum0 = _mm256_setzero_ps(), sum1 = _mm256_setzero_ps();
  const float *end = x + d;
  while (x < end) {
    {
      auto xx = _mm256_loadu_ps(x);
      x += 8;
      auto zz = _mm_loadu_si128((__m128i *)y);
      y += 8;
      auto yy = _mm256_cvtepu16_epi32(zz);
      yy = _mm256_slli_epi32(yy, 16);
      sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(xx, (__m256)yy));
    }
    {
      auto xx = _mm256_loadu_ps(x);
      x += 8;
      auto zz = _mm_loadu_si128((__m128i *)y);
      y += 8;
      auto yy = _mm256_cvtepu16_epi32(zz);
      yy = _mm256_slli_epi32(yy, 16);
      sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(xx, (__m256)yy));
    }
  }
  sum1 = _mm256_add_ps(sum1, sum0);
  auto sumh =
      _mm_add_ps(_mm256_castps256_ps128(sum1), _mm256_extractf128_ps(sum1, 1));
  auto tmp1 = _mm_add_ps(sumh, _mm_movehl_ps(sumh, sumh));
  auto tmp2 = _mm_add_ps(tmp1, _mm_movehdup_ps(tmp1));
  return _mm_cvtss_f32(tmp2);
}

}  // namespace bf16

namespace bf16_avx512 {

uint16_t *code, *q_code;

void FloatToBFloat16(const float *src, void *dst, int64_t size) {
  const uint16_t *p = reinterpret_cast<const uint16_t *>(src);
  uint16_t *q = reinterpret_cast<uint16_t *>(dst);
  for (; size != 0; p += 2, q++, size--) {
    *q = p[1];
  }
}

void init() {
  code = (uint16_t *)aligned_alloc(64, N * dim * 2);
  q_code = (uint16_t *)aligned_alloc(64, dim * 2);
  for (int i = 0; i < N; ++i) {
    FloatToBFloat16(data + i * dim, code + i * dim, dim);
  }
  FloatToBFloat16(query, q_code, dim);
}

float IPAvx512(const int idx, const int d) {
  const uint16_t *x = q_code;
  const uint16_t *y = code + idx * d;
  __m512 sum0 = _mm512_setzero_ps(), sum1 = _mm512_setzero_ps();
  const uint16_t *end = x + d;
  while (x < end) {
    {
      auto xx = (__m512bh)_mm512_loadu_si512(x);
      x += 32;
      auto yy = (__m512bh)_mm512_loadu_si512(y);
      y += 32;
      sum0 = _mm512_dpbf16_ps(sum0, xx, yy);
    }
    {
      auto xx = (__m512bh)_mm512_loadu_si512(x);
      x += 32;
      auto yy = (__m512bh)_mm512_loadu_si512(y);
      y += 32;
      sum1 = _mm512_dpbf16_ps(sum1, xx, yy);
    }
  }
  sum1 = _mm512_add_ps(sum1, sum0);
  auto sumhh = _mm256_add_ps(_mm512_castps512_ps256(sum1),
                             _mm512_extractf32x8_ps(sum1, 1));
  auto sumh = _mm_add_ps(_mm256_castps256_ps128(sumhh),
                         _mm256_extractf128_ps(sumhh, 1));
  auto tmp1 = _mm_add_ps(sumh, _mm_movehl_ps(sumh, sumh));
  auto tmp2 = _mm_add_ps(tmp1, _mm_movehdup_ps(tmp1));
  return _mm_cvtss_f32(tmp2);
}
}  // namespace bf16_avx512

static void Naive(benchmark::State &state) {
  // Code inside this loop is measured repeatedly
  for (int i = 0; auto _ : state) {
    ++i %= N;
    float dist = naive::IP(perm[i], dim);
    benchmark::DoNotOptimize(dist);
  }
}

static void FaissIP(benchmark::State &state) {
  // Code inside this loop is measured repeatedly
  for (int i = 0; auto _ : state) {
    ++i %= N;
    float dist = faiss::IP(perm[i], dim);
    benchmark::DoNotOptimize(dist);
  }
}

static void FastIP(benchmark::State &state) {
  // Code inside this loop is measured repeatedly
  for (int i = 0; auto _ : state) {
    ++i %= N;
    float dist = fast::IP(perm[i], dim);
    benchmark::DoNotOptimize(dist);
  }
}

static void FastIPAvx512(benchmark::State &state) {
  // Code inside this loop is measured repeatedly
  for (int i = 0; auto _ : state) {
    ++i %= N;
    float dist = fast_avx512::IP(perm[i], dim);
    benchmark::DoNotOptimize(dist);
  }
}

static void BF16(benchmark::State &state) {
  // Code inside this loop is measured repeatedly
  for (int i = 0; auto _ : state) {
    ++i %= N;
    float dist = bf16::IP(perm[i], dim);
    benchmark::DoNotOptimize(dist);
  }
}

static void BF16Avx512(benchmark::State &state) {
  // Code inside this loop is measured repeatedly
  for (int i = 0; auto _ : state) {
    ++i %= N;
    float dist = bf16_avx512::IPAvx512(perm[i], dim);
    benchmark::DoNotOptimize(dist);
  }
}

int main(int argc, char **argv) {
  N = std::stoll(argv[1]);
  dim = std::stoll(argv[2]);
  global_init();
  std::cout << "global init done" << std::endl;
  checker::init();
  std::cout << "checker init done" << std::endl;
  // BENCHMARK(Naive);
  checker::check(faiss::IP);
  BENCHMARK(FaissIP);
  checker::check(fast::IP);
  BENCHMARK(FastIP);
  checker::check(fast_avx512::IP);
  BENCHMARK(FastIPAvx512);
  bf16::init();
  checker::check(bf16::IP);
  BENCHMARK(BF16);
  bf16_avx512::init();
  checker::check(bf16_avx512::IPAvx512);
  BENCHMARK(BF16Avx512);
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  global_destroy();
}
