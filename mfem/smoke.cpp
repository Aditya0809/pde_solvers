#include "mfem.hpp"
#include <mpi.h>
#include <iostream>

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  int rank=0, size=0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if(rank==0) std::cout << "MFEM smoke test OK, size=" << size << "\n";

  MPI_Finalize();
  return 0;
}