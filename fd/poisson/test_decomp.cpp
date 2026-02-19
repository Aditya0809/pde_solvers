#include <mpi.h>
#include <cstdio>
#include <cmath>
#include "decomp2d.hpp"
#include<vector>
#include<unordered_set>
#include<utility>



int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int  Nx = 100, Ny = 100;
  int Px = 4, Py = 3;
  int nghost = 1;

  Decomp2D decomp(MPI_COMM_WORLD, Nx, Ny, Px, Py, nghost); // Example: global grid 100x100, process grid sqrt(size) x sqrt(size)

  float hx = 1.0 / (decomp.Nx() - 1);
  float hy = 1.0 / (decomp.Ny() - 1);

  int nghost = decomp.nghost();
  int nx = decomp.nx(), ny = decomp.ny();

  const std::size_t local_size_x = nx + 2*nghost;
  const std::size_t local_size_y = ny + 2*nghost;

  std::vector<float> u(local_size_x * local_size_y);
  std::vector<float> u_new(local_size_x * local_size_y);
  std::vector<float> f(local_size_x * local_size_y);

  auto index = [&](std::size_t i, std::size_t j) {
    return i * local_size_y + j; // Assuming row-major order
  };



  for(std::size_t i = 0; i < decomp.nx(); ++i) {
    for(std::size_t j = 0; j < decomp.ny(); ++j) {
      int global_i = decomp.i0() + i;
      int global_j = decomp.j0() + j;
      float x = global_i * hx;
      float y = global_j * hy;
      f[index(i+nghost,j+nghost)] = 2.0 * M_PI * M_PI * std::sin(M_PI * x) * std::sin(M_PI * y); // Example source term}
      u[index(i+nghost,j+nghost)] = 0.0; // Initial guess
    }
  }

  // Halo exchange

  // Example: exchange with left and right neighbors
  int left = decomp.left();
  int right = decomp.right();
  int up = decomp.up();
  int down = decomp.down();

  std::vector<float> send_column(nghost*ny);
  std::vector<float> recv_column(nghost*ny);
  std::vector<float> send_row(nghost*nx);
  std::vector<float> recv_row(nghost*nx);

  if(left != MPI_PROC_NULL && nghost > 0) {
    
    


    for(std::size_t ii=0; ii < nghost; ++ii) {
      // Send left ghost layer
      MPI_Send(&u[index(decomp.nghost(), decomp.nghost() + ii)], decomp.ny(), MPI_FLOAT, left, rank, MPI_COMM_WORLD);
      // Receive right ghost layer
      MPI_Recv(&u[index(decomp.nghost(), ii)], decomp.ny(), MPI_FLOAT,left, rank+1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }






  

  MPI_Finalize();
  return 0;
}