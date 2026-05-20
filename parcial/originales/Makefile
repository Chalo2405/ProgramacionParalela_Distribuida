BREW_GCC := $(shell command -v gcc-15 2>/dev/null)

CC := $(if $(BREW_GCC),$(BREW_GCC),$(CC))
MPICC ?= mpicc
CFLAGS ?= -O2 -Wall -Wextra
OPENMP_FLAGS ?= -fopenmp

.PHONY: all openmp mpi clean

all: openmp mpi

openmp: pi_leibniz_openmp_original pi_montecarlo_openmp_original

mpi: reduce_avg_mpi_original reduce_stddev_mpi_original

pi_leibniz_openmp_original: pi_leibniz_openmp_original.c
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) $< -o $@

pi_montecarlo_openmp_original: pi_montecarlo_openmp_original.c
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) $< -o $@

reduce_avg_mpi_original: reduce_avg_mpi_original.c
	$(MPICC) $(CFLAGS) $< -o $@

reduce_stddev_mpi_original: reduce_stddev_mpi_original.c
	$(MPICC) $(CFLAGS) $< -o $@ -lm

clean:
	rm -f pi_leibniz_openmp_original pi_montecarlo_openmp_original
	rm -f reduce_avg_mpi_original reduce_stddev_mpi_original
