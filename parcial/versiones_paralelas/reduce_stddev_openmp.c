#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static float random01_from_index(long long index) {
  unsigned int state = (unsigned int)(index + 1) * 747796405u + 2891336453u;
  state = state * 1664525u + 1013904223u;
  return (float)state / 4294967295.0f;
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
  double global_sq_diff = 0.0;
  int num_threads = 0;

  double tiempo_inicio = omp_get_wtime();

#pragma omp parallel reduction(+ : global_sum)
  {
    int tid = omp_get_thread_num();
    if (tid == 0) {
      num_threads = omp_get_num_threads();
    }

#pragma omp for
    for (long long i = 0; i < num_elements; i++) {
      global_sum += random01_from_index(i);
    }
  }

  double mean = global_sum / (double)num_elements;

#pragma omp parallel for reduction(+ : global_sq_diff)
  for (long long i = 0; i < num_elements; i++) {
    double diff = (double)random01_from_index(i) - mean;
    global_sq_diff += diff * diff;
  }

  double stddev = sqrt(global_sq_diff / (double)num_elements);
  double tiempo_final = omp_get_wtime();

  printf("Ejecucion OpenMP con %d threads\n", num_threads);
  printf("Mean - %f, Standard deviation = %f\n", mean, stddev);
  printf("Tiempo de ejecucion: %f segundos\n",
         tiempo_final - tiempo_inicio);

  return 0;
}
