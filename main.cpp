#include <array>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
//
#ifdef __AVX__
#include <immintrin.h>
#else
#error "No AVX supported!"
#endif

using namespace std;

template <typename T, typename U>
inline auto pun_cast(U u) {
  static_assert(sizeof(T) == sizeof(U), "");
  T result;
  std::memcpy(&result, &u, sizeof(U));
  return result;
}

inline ostream& operator<<(ostream& os, __m256 v) {
  const auto out = pun_cast<array<float, 8>>(v);
  for (auto x : out) os << setw(12) << x;
  return os << '\n';
}

int main() {
  auto v = _mm256_set_ps(1, 2, 3, 4, 5, 6, 7, 8);
  auto w = _mm256_set_ps(1, 2, 1, 2, 1, 2, 1, 2);
  auto v2 = _mm256_mul_ps(v, v);
  auto w2 = _mm256_mul_ps(w, w);
  auto r2 = _mm256_add_ps(v2, w2);
  auto r = _mm256_sqrt_ps(r2);
  cout << v << w << v2 << w2 << r2 << r;

  const size_t samples = 100'000'000;

  mt19937 rng{random_device{}()};
  uniform_real_distribution<float> distribution{0, 1};
  {
    const auto start = chrono::high_resolution_clock::now();
    size_t count = 0;
    for (size_t i = 0; i < samples; ++i) {
      const auto x = distribution(rng);
      const auto y = distribution(rng);
      count += (x * x + y * y <= 1);
    }
    const auto monte_carlo_pi = 4.0f * count / samples;
    const auto end = chrono::high_resolution_clock::now();
    const auto time = chrono::duration<float>(end - start).count();

    cout << "Monte Carlo pi = " << monte_carlo_pi << '\n'
         << "time = " << time << " s\n";
  }
  {
    const auto start = chrono::high_resolution_clock::now();

    auto count8 = _mm256_setzero_ps();
    for (size_t i = 0; i < samples; i += 8) {
      const auto x =
          _mm256_set_ps(distribution(rng), distribution(rng), distribution(rng),
                        distribution(rng), distribution(rng), distribution(rng),
                        distribution(rng), distribution(rng));
      const auto y =
          _mm256_set_ps(distribution(rng), distribution(rng), distribution(rng),
                        distribution(rng), distribution(rng), distribution(rng),
                        distribution(rng), distribution(rng));
      const auto x2 = _mm256_mul_ps(x, x);
      const auto y2 = _mm256_mul_ps(y, y);
      const auto r2 = _mm256_add_ps(x2, y2);
      const auto mask = _mm256_cmp_ps(r2, _mm256_set1_ps(1.0f), _CMP_LE_OQ);
      const auto inc =
          _mm256_blendv_ps(_mm256_setzero_ps(), _mm256_set1_ps(1.0f), mask);
      count8 = _mm256_add_ps(count8, inc);
    }

    const auto tmp = pun_cast<array<float, 8>>(count8);
    float count =
        tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    const auto monte_carlo_pi = 4.0f * count / samples;
    const auto end = chrono::high_resolution_clock::now();
    const auto time = chrono::duration<float>(end - start).count();

    cout << "Monte Carlo pi = " << monte_carlo_pi << '\n'
         << "time = " << time << " s\n";
  }
}