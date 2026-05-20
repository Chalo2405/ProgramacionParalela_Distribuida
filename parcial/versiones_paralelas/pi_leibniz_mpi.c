#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  int rank, size;
  long long numero_iteraciones = 0;
  double suma_local = 0.0;
  double respuesta = 0.0;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    if (argc >= 2) {
      numero_iteraciones = atoll(argv[1]);
    } else {
      printf("Ingresar el numero de iteraciones: ");
      fflush(stdout);
      if (scanf("%lld", &numero_iteraciones) != 1) {
        fprintf(stderr, "Entrada invalida.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
    }
  }

  MPI_Bcast(&numero_iteraciones, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

  if (numero_iteraciones <= 0) {
    if (rank == 0) {
      fprintf(stderr, "El numero de iteraciones debe ser mayor que cero.\n");
    }
    MPI_Finalize();
    return 1;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double tiempo_inicio = MPI_Wtime();

  for (long long indice = rank; indice < numero_iteraciones; indice += size) {
    double termino = 4.0 / (2.0 * (double)indice + 1.0);
    suma_local += (indice % 2 == 0) ? termino : -termino;
  }

  MPI_Reduce(&suma_local, &respuesta, 1, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_WORLD);

  double tiempo_final = MPI_Wtime();

  if (rank == 0) {
    printf("Ejecucion MPI con %d procesos\n", size);
    printf("La respuesta es: %.8f\n", respuesta);
    printf("Tiempo de ejecucion: %f segundos\n",
           tiempo_final - tiempo_inicio);
  }

  MPI_Finalize();
  return 0;
}
