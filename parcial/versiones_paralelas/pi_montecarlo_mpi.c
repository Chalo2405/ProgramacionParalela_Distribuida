#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {
  int rank, size;
  long long samples = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (argc != 2) {
    if (rank == 0) {
      fprintf(stderr, "Uso: %s samples\n", argv[0]);
    }
    MPI_Finalize();
    return 1;
  }

  samples = atoll(argv[1]);
  if (samples <= 0) {
    if (rank == 0) {
      fprintf(stderr, "samples debe ser mayor que cero.\n");
    }
    MPI_Finalize();
    return 1;
  }

  long long base = samples / size;
  long long remainder = samples % size;
  long long local_samples = base + (rank < remainder ? 1 : 0);
  long long local_count = 0;

  unsigned short seed[3];
  unsigned int now = (unsigned int)time(NULL);
  seed[0] = (unsigned short)(0x330e + rank);
  seed[1] = (unsigned short)(now ^ (rank * 0x9e37u));
  seed[2] = (unsigned short)(rank + 1);

  MPI_Barrier(MPI_COMM_WORLD);
  double tiempo_inicio = MPI_Wtime();

  for (long long i = 0; i < local_samples; i++) {
    double x = erand48(seed);
    double y = erand48(seed);
    if (x * x + y * y <= 1.0) {
      local_count++;
    }
  }

  long long total_count = 0;
  MPI_Reduce(&local_count, &total_count, 1, MPI_LONG_LONG, MPI_SUM, 0,
             MPI_COMM_WORLD);

  double tiempo_final = MPI_Wtime();

  if (rank == 0) {
    double pi = 4.0 * (double)total_count / (double)samples;
    printf("Ejecucion MPI con %d procesos\n", size);
    printf("Count = %lld, Samples = %lld, Estimate of pi: %7.5f\n",
           total_count, samples, pi);
    printf("Tiempo de ejecucion: %f segundos\n",
           tiempo_final - tiempo_inicio);
  }

  MPI_Finalize();
  return 0;
}
