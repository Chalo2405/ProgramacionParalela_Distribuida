#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                                        \
  do {                                                                         \
    cudaError_t status = (call);                                                \
    if (status != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(status));                                      \
      exit(EXIT_FAILURE);                                                       \
    }                                                                          \
  } while (0)

__device__ uint32_t lcg_next(uint32_t *state) {
  *state = (*state * 1664525u + 1013904223u);
  return *state;
}

__device__ float random01(uint32_t *state) {
  return (float)lcg_next(state) / 4294967295.0f;
}

__global__ void avg_kernel(long long n, double *block_sums) {
  extern __shared__ double partial[];
  unsigned int tid = threadIdx.x;
  long long global_id = blockIdx.x * blockDim.x + threadIdx.x;
  long long stride = (long long)gridDim.x * blockDim.x;
  uint32_t state = (uint32_t)(global_id + 1) * 747796405u;
  double sum = 0.0;

  for (long long i = global_id; i < n; i += stride) {
    sum += random01(&state);
  }

  partial[tid] = sum;
  __syncthreads();

  for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      partial[tid] += partial[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    block_sums[blockIdx.x] = partial[0];
  }
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "Uso: %s num_elements\n", argv[0]);
    return 1;
  }

  long long num_elements = atoll(argv[1]);
  if (num_elements <= 0) {
    fprintf(stderr, "num_elements debe ser mayor que cero.\n");
    return 1;
  }

  const int threads_per_block = 256;
  int blocks = (int)((num_elements + threads_per_block - 1) /
                     threads_per_block);
  if (blocks < 1) {
    blocks = 1;
  }
  if (blocks > 65535) {
    blocks = 65535;
  }

  double *d_block_sums = NULL;
  double *h_block_sums = (double *)malloc(sizeof(double) * blocks);
  if (h_block_sums == NULL) {
    fprintf(stderr, "No se pudo reservar memoria en CPU.\n");
    return 1;
  }

  CUDA_CHECK(cudaMalloc((void **)&d_block_sums, sizeof(double) * blocks));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  avg_kernel<<<blocks, threads_per_block,
               sizeof(double) * threads_per_block>>>(num_elements,
                                                     d_block_sums);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  CUDA_CHECK(cudaMemcpy(h_block_sums, d_block_sums, sizeof(double) * blocks,
                        cudaMemcpyDeviceToHost));

  double global_sum = 0.0;
  for (int i = 0; i < blocks; i++) {
    global_sum += h_block_sums[i];
  }

  float milliseconds = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

  printf("Ejecucion CUDA con %d bloques y %d hilos por bloque\n", blocks,
         threads_per_block);
  printf("Total sum = %f, avg = %f\n", global_sum,
         global_sum / (double)num_elements);
  printf("Tiempo de kernel: %f segundos\n", milliseconds / 1000.0f);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_block_sums));
  free(h_block_sums);

  return 0;
}
