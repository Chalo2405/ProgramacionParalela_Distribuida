#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static float random01(unsigned int *seed) {
  *seed = (*seed * 1103515245u + 12345u);
  return (float)((*seed / 65536u) % 32768u) / 32767.0f;
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

  double global_sum = 0.0;
  int num_threads = 0;

  double tiempo_inicio = omp_get_wtime();

#pragma omp parallel reduction(+ : global_sum)
  {
    int tid = omp_get_thread_num();
    int threads = omp_get_num_threads();
    long long base = num_elements / threads;
    long long remainder = num_elements % threads;
    long long local_n = base + (tid < remainder ? 1 : 0);
    long long start = tid * base + (tid < remainder ? tid : remainder);
    unsigned int seed = (unsigned int)time(NULL) ^ (unsigned int)(tid + 1);
    double local_sum = 0.0;

    if (tid == 0) {
      num_threads = threads;
    }

    for (long long i = 0; i < local_n; i++) {
      unsigned int local_seed = seed + (unsigned int)(start + i);
      local_sum += random01(&local_seed);
    }

    global_sum += local_sum;

#pragma omp critical
    {
      printf("Local sum for thread %d - %f, avg = %f\n", tid, local_sum,
             local_n > 0 ? local_sum / (double)local_n : 0.0);
    }
  }

  double tiempo_final = omp_get_wtime();

  printf("Ejecucion OpenMP con %d threads\n", num_threads);
  printf("Total sum = %f, avg = %f\n", global_sum,
         global_sum / (double)num_elements);
  printf("Tiempo de ejecucion: %f segundos\n",
         tiempo_final - tiempo_inicio);

  return 0;
}
