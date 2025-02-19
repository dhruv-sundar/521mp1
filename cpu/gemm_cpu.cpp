#include "../include/utils.h"
#include <chrono>

#define NUM_RUNS 2

#define CHECK(name)                                                            \
  std::cout << "checking " << #name << std::endl;                              \
  initialize(refC, Ref::M *Ref::N);                                            \
  name(ref.A, ref.B, refC, Ref::M, Ref::N, Ref::K);                            \
  if (!ref.checkRef(refC)) {                                                   \
    std::cerr << #name << ": check ref failed!" << std::endl;                  \
  };

#define TIME(name)                                                             \
  for (int i = 0; i < 1; i++) {                                                \
    name(A, B, C, M, N, K);                                                    \
  }                                                                            \
  std::chrono::duration<double, std::milli> time_##name;                       \
  for (int i = 0; i < NUM_RUNS; i++) {                                         \
    initialize(C, M *N);                                                       \
    auto start_time_##name = std::chrono::high_resolution_clock::now();        \
    name(A, B, C, M, N, K);                                                    \
    auto end_time_##name = std::chrono::high_resolution_clock::now();          \
    time_##name += end_time_##name - start_time_##name;                        \
  }                                                                            \
  std::chrono::duration<double, std::milli> duration_##name =                  \
      time_##name / float(NUM_RUNS);                                           \
  std::cout << "Time taken for GEMM (CPU," << #name                            \
            << "): " << duration_##name.count() << "ms" << std::endl;

// We need to fit:
// 1. One tile of B (TILE_SIZE × TILE_SIZE)
// 2. One row of C (TILE_SIZE)
// 3. One element of A (1)
// into L1 cache (32KB)
// Required elements: 1 + TILE_SIZE + TILE_SIZE * TILE_SIZE = 32KB
// So TILE_SIZE + TILE_SIZE^2 = 32KB / 4B = 8192
// TILE_SIZE^2 + TILE_SIZE - 8192 = 0
// TILE_SIZE = (-1 + sqrt(1 + 4 * 8192)) / 2 = 90 ish

#define TILE_SIZE 16

// reference CPU implementation of the GEMM kernel
// note that this implementation is naive and will run for longer for larger
// graphs
void gemm_cpu_o0(float *A, float *B, float *C, int M, int N, int K) {
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < M; i++) {
      for (int k = 0; k < K; k++) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}

// Your optimized implementations go here
// note that for o4 you don't have to change the code, but just the compiler
// flags. So, you can use o3's code for that part
void gemm_cpu_o1(float *A, float *B, float *C, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      const float A_ik = A[i * K + k];
      for (int j = 0; j < N; j++) {
        C[i * N + j] += A_ik * B[k * N + j];
      }
    }
  }
}

void gemm_cpu_o2(float *A, float *B, float *C, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int kt = 0; kt < K; kt += TILE_SIZE) {
      for (int jt = 0; jt < N; jt += TILE_SIZE) {

        const int k_end = std::min(kt + TILE_SIZE, K);
        const int j_end = std::min(jt + TILE_SIZE, N);

        // Process one tile
        for (int k = kt; k < k_end; k++) {
          const float A_ik = A[i * K + k];
          for (int j = jt; j < j_end; j++) {
            C[i * N + j] += A_ik * B[k * N + j];
          }
        }
      }
    }
  }
}

void gemm_cpu_o3(float *A, float *B, float *C, int M, int N, int K) {
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < M; i++) {
    for (int kt = 0; kt < K; kt += TILE_SIZE) {
      for (int jt = 0; jt < N; jt += TILE_SIZE) {
        const int k_end = std::min(kt + TILE_SIZE, K);
        const int j_end = std::min(jt + TILE_SIZE, N);

        // Process one tile
        for (int k = kt; k < k_end; k++) {
          const float A_ik = A[i * K + k];
#pragma omp simd
          for (int j = jt; j < j_end; j++) {
            C[i * N + j] += A_ik * B[k * N + j];
          }
        }
      }
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cout << "Usage: mp1 <M> <N> <K>" << std::endl;
    return 1;
  }

  int M = atoi(argv[1]);
  int N = atoi(argv[2]);
  int K = atoi(argv[3]);

  float *A = new float[M * K]();
  float *B = new float[K * N]();
  float *C = new float[M * N]();

  fillRandom(A, M * K);
  fillRandom(B, K * N);

  // Check if the kernel results are correct
  // note that even if the correctness check fails all optimized kernels will
  // run. We are not exiting the program at failure at this point. It is a
  // good idea to add more correctness checks to your code. We may (at
  // discretion) verify that your code is correct.
  float *refC = new float[Ref::M * Ref::N]();
  auto ref = Ref();
  CHECK(gemm_cpu_o0)
  CHECK(gemm_cpu_o1)
  CHECK(gemm_cpu_o2)
  CHECK(gemm_cpu_o3)
  delete[] refC;

  TIME(gemm_cpu_o0)
  TIME(gemm_cpu_o1)
  TIME(gemm_cpu_o2)
  TIME(gemm_cpu_o3)

  delete[] A;
  delete[] B;
  delete[] C;

  return 0;
}
