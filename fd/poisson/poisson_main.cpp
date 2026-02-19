#include <mpi.h>
#include <cstdio>
#include <cmath>
#include "decomp2d.hpp"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  //std::printf("poisson_fd built with MPI. ranks=%d/size=%d\n", rank, size);

  Decomp2D decomp(MPI_COMM_WORLD, 100, 100, 1, 3); // Example: global grid 100x100, process grid sqrt(size) x sqrt(size)

  std::printf("Rank %d: local grid bounds i=[%d, %d), j=[%d, %d), px=%d, py=%d, neighbors (left=%d, right=%d, up=%d, down=%d)\n",
              decomp.rank(), decomp.i0(), decomp.i1(), decomp.j0(), decomp.j1(),
              decomp.px(), decomp.py(), decomp.left(), decomp.right(), decomp.up(), decomp.down());

  MPI_Finalize();
  return 0;
}
