#include <cuda_runtime.h>
#include <math.h>
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

__device__ float random01_from_index(long long index) {
  uint32_t state = (uint32_t)(index + 1) * 747796405u + 2891336453u;
  state = state * 1664525u + 1013904223u;
  return (float)state / 4294967295.0f;
}

__global__ void sum_kernel(long long n, double *block_sums) {
  extern __shared__ double partial[];
  unsigned int tid = threadIdx.x;
  long long global_id = blockIdx.x * blockDim.x + threadIdx.x;
  long long stride = (long long)gridDim.x * blockDim.x;
  double sum = 0.0;

  for (long long i = global_id; i < n; i += stride) {
    sum += random01_from_index(i);
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

__global__ void sq_diff_kernel(long long n, double mean,
                               double *block_sq_diffs) {
  extern __shared__ double partial[];
  unsigned int tid = threadIdx.x;
  long long global_id = blockIdx.x * blockDim.x + threadIdx.x;
  long long stride = (long long)gridDim.x * blockDim.x;
  double sum = 0.0;

  for (long long i = global_id; i < n; i += stride) {
    double diff = (double)random01_from_index(i) - mean;
    sum += diff * diff;
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
    block_sq_diffs[blockIdx.x] = partial[0];
  }
}

static double sum_blocks(const double *values, int blocks) {
  double total = 0.0;
  for (int i = 0; i < blocks; i++) {
    total += values[i];
  }
  return total;
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

  double *d_block_values = NULL;
  double *h_block_values = (double *)malloc(sizeof(double) * blocks);
  if (h_block_values == NULL) {
    fprintf(stderr, "No se pudo reservar memoria en CPU.\n");
    return 1;
  }

  CUDA_CHECK(cudaMalloc((void **)&d_block_values, sizeof(double) * blocks));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  sum_kernel<<<blocks, threads_per_block,
               sizeof(double) * threads_per_block>>>(num_elements,
                                                     d_block_values);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(h_block_values, d_block_values, sizeof(double) * blocks,
                        cudaMemcpyDeviceToHost));

  double mean = sum_blocks(h_block_values, blocks) / (double)num_elements;

  sq_diff_kernel<<<blocks, threads_per_block,
                   sizeof(double) * threads_per_block>>>(num_elements, mean,
                                                         d_block_values);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(h_block_values, d_block_values, sizeof(double) * blocks,
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  double global_sq_diff = sum_blocks(h_block_values, blocks);
  double stddev = sqrt(global_sq_diff / (double)num_elements);

  float milliseconds = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

  printf("Ejecucion CUDA con %d bloques y %d hilos por bloque\n", blocks,
         threads_per_block);
  printf("Mean - %f, Standard deviation = %f\n", mean, stddev);
  printf("Tiempo de kernels: %f segundos\n", milliseconds / 1000.0f);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_block_values));
  free(h_block_values);

  return 0;
}
