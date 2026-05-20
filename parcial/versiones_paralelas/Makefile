BREW_GCC := $(shell command -v gcc-15 2>/dev/null)

CC := $(if $(BREW_GCC),$(BREW_GCC),$(CC))
MPICC ?= mpicc
NVCC ?= nvcc
OPENMP_FLAGS ?= -fopenmp
CFLAGS ?= -O2 -Wall -Wextra
LDFLAGS_MATH ?= -lm
CUDAFLAGS ?= -O2

.PHONY: all openmp mpi cuda clean

all: openmp mpi

openmp: reduce_avg_openmp reduce_stddev_openmp

mpi: pi_leibniz_mpi pi_montecarlo_mpi

cuda: pi_leibniz_cuda pi_montecarlo_cuda reduce_avg_cuda reduce_stddev_cuda

pi_leibniz_mpi: pi_leibniz_mpi.c
	$(MPICC) $(CFLAGS) $< -o $@

pi_montecarlo_mpi: pi_montecarlo_mpi.c
	$(MPICC) $(CFLAGS) $< -o $@

reduce_avg_openmp: reduce_avg_openmp.c
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) $< -o $@

reduce_stddev_openmp: reduce_stddev_openmp.c
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) $< -o $@ $(LDFLAGS_MATH)

pi_leibniz_cuda: pi_leibniz_cuda.cu
	$(NVCC) $(CUDAFLAGS) $< -o $@

pi_montecarlo_cuda: pi_montecarlo_cuda.cu
	$(NVCC) $(CUDAFLAGS) $< -o $@

reduce_avg_cuda: reduce_avg_cuda.cu
	$(NVCC) $(CUDAFLAGS) $< -o $@

reduce_stddev_cuda: reduce_stddev_cuda.cu
	$(NVCC) $(CUDAFLAGS) $< -o $@

clean:
	rm -f pi_leibniz_mpi pi_montecarlo_mpi
	rm -f reduce_avg_openmp reduce_stddev_openmp
	rm -f pi_leibniz_cuda pi_montecarlo_cuda reduce_avg_cuda reduce_stddev_cuda
