#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CUDA_CHECK(call)                                                        \
  do {                                                                         \
    cudaError_t status = (call);                                                \
    if (status != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(status));                                      \
      exit(EXIT_FAILURE);                                                       \
    }                                                                          \
  } while (0)

__device__ uint32_t xorshift32(uint32_t *state) {
  uint32_t x = *state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *state = x;
  return x;
}

__device__ double random01(uint32_t *state) {
  return (double)xorshift32(state) / 4294967296.0;
}

__global__ void montecarlo_kernel(long long samples, unsigned int seed,
                                  unsigned long long *block_counts) {
  extern __shared__ unsigned int partial[];
  unsigned int tid = threadIdx.x;
  long long global_id = blockIdx.x * blockDim.x + threadIdx.x;
  long long stride = (long long)gridDim.x * blockDim.x;
  uint32_t state = seed ^ (uint32_t)(global_id + 1) * 747796405u;
  unsigned int count = 0;

  for (long long i = global_id; i < samples; i += stride) {
    double x = random01(&state);
    double y = random01(&state);
    if (x * x + y * y <= 1.0) {
      count++;
    }
  }

  partial[tid] = count;
  __syncthreads();

  for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      partial[tid] += partial[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    block_counts[blockIdx.x] = partial[0];
  }
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "Uso: %s samples\n", argv[0]);
    return 1;
  }

  long long samples = atoll(argv[1]);
  if (samples <= 0) {
    fprintf(stderr, "samples debe ser mayor que cero.\n");
    return 1;
  }

  const int threads_per_block = 256;
  int blocks = (int)((samples + threads_per_block - 1) / threads_per_block);
  if (blocks < 1) {
    blocks = 1;
  }
  if (blocks > 65535) {
    blocks = 65535;
  }

  unsigned long long *d_block_counts = NULL;
  unsigned long long *h_block_counts =
      (unsigned long long *)malloc(sizeof(unsigned long long) * blocks);
  if (h_block_counts == NULL) {
    fprintf(stderr, "No se pudo reservar memoria en CPU.\n");
    return 1;
  }

  CUDA_CHECK(cudaMalloc((void **)&d_block_counts,
                        sizeof(unsigned long long) * blocks));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  montecarlo_kernel<<<blocks, threads_per_block,
                      sizeof(unsigned int) * threads_per_block>>>(
      samples, (unsigned int)time(NULL), d_block_counts);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  CUDA_CHECK(cudaMemcpy(h_block_counts, d_block_counts,
                        sizeof(unsigned long long) * blocks,
                        cudaMemcpyDeviceToHost));

  unsigned long long total_count = 0;
  for (int i = 0; i < blocks; i++) {
    total_count += h_block_counts[i];
  }

  double pi = 4.0 * (double)total_count / (double)samples;
  float milliseconds = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

  printf("Ejecucion CUDA con %d bloques y %d hilos por bloque\n", blocks,
         threads_per_block);
  printf("Count = %llu, Samples = %lld, Estimate of pi: %7.5f\n",
         total_count, samples, pi);
  printf("Tiempo de kernel: %f segundos\n", milliseconds / 1000.0f);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_block_counts));
  free(h_block_counts);

  return 0;
}
