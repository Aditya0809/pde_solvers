#include <mpi.h>
#include <cstdio>
#include "decomp2d.hpp"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::printf("poisson_fd built with MPI. ranks=%d/size=%d\n", rank, size);

  MPI_Finalize();
  return 0;
}
