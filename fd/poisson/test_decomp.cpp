#include <mpi.h>
#include <cstdio>
#include <cmath>
#include "decomp2d.hpp"
#include<vector>
#include<utility>
#include "haloExchange.hpp"



int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int  Nx = 128, Ny = 128;
  int Px = 4, Py = 4;
  int nghost = 1;

  Decomp2D decomp(MPI_COMM_WORLD, Nx, Ny, Px, Py, nghost); // Example: global grid 100x100, process grid sqrt(size) x sqrt(size)
  HaloExchange halo_exchange(decomp);

  
  float hx = 1.0 / (decomp.Nx() - 1);
  float hy = 1.0 / (decomp.Ny() - 1);

  if (rank == 0) {
    std::printf("Nx Ny = %d %d | decomp.Nx Ny = %d %d | hx hy = %.6g %.6g\n",
          Nx, Ny, decomp.Nx(), decomp.Ny(), hx, hy);
  }

  int ng = decomp.nghost();
  int nx = decomp.nx(), ny = decomp.ny();

  const int local_size_x = nx + 2*ng;
  const int local_size_y = ny + 2*ng;

  std::vector<float> u(local_size_x * local_size_y);
  std::vector<float> u_new(local_size_x * local_size_y);
  std::vector<float> f(local_size_x * local_size_y);

  // fill f and initial guess for u
  std::fill(u.begin(), u.end(), 0.0f);
  std::fill(u_new.begin(), u_new.end(), 0.0f);
  std::fill(f.begin(), f.end(), 0.0f);


  auto index = [&](int i, int j) {
    return i * local_size_y + j; // Assuming row-major order
  };

  auto local_to_global = [&](int i, int j) {
    int global_i = decomp.i0() + i;
    int global_j = decomp.j0() + j;
    return std::make_pair(global_i, global_j);
  };



  for(int i = 0; i < nx; ++i) {
    for(int j = 0; j < ny; ++j) {
      auto [global_i, global_j] = local_to_global(i, j);
      float x = global_i * hx;
      float y = global_j * hy;
      f[index(i+ng,j+ng)] = 2.0 * M_PI * M_PI * std::sin(M_PI * x) * std::sin(M_PI * y); // Example source term}
      u[index(i+ng,j+ng)] = 0.0; // Initial guess
    }
  }

  // Halo exchange

  // // Example: exchange with left and right neighbors
  // int left = decomp.left();
  // int right = decomp.right();
  // int up = decomp.up();
  // int down = decomp.down();

  // std::vector<float> send_column_left(ng*ny);
  // std::vector<float> recv_column_left(ng*ny);
  // std::vector<float> send_column_right(ng*ny);
  // std::vector<float> recv_column_right(ng*ny);
  // std::vector<float> send_row_top(ng*nx);
  // std::vector<float> recv_row_top(ng*nx);
  // std::vector<float> send_row_bottom(ng*nx);
  // std::vector<float> recv_row_bottom(ng*nx);

  // // initiate tags left, right, up, down
  // const int tag_x_l2r = 10; // tag for left to right communication
  // const int tag_x_r2l = 11; // tag for right to left communication
  // const int tag_y_b2t = 12; // tag for bottom to top communication
  // const int tag_y_t2b = 13; // tag for top to bottom communication

  // auto halo_exchange = [&](std::vector<float> &U) -> void {
    

  //   for(int g=0; g < ng; ++g) {
  //     // Prepare left and rigtht ghost layer to send
  //     for(int j=0; j < ny; ++j) {
  //       send_column_left[g*ny + j] = U[index(ng+g, ng+j)]; // left ghost layer
  //       send_column_right[g*ny + j] = U[index(ng + nx -ng  + g, ng+j)]; // right ghost layer
  //     }
  //     // Prepare top and bottom ghost layer to send
  //     for(int i=0; i < nx; ++i) {
  //       send_row_bottom[g*nx + i] = U[index(ng+i, ng+g)]; // bottom ghost layer
  //       send_row_top[g*nx + i] = U[index(ng+i, ng + ny - ng + g)]; // top ghost layer
  //     }
  //   }


    

  //   if(left != MPI_PROC_NULL && ng > 0) {
  //     MPI_Sendrecv(send_column_left.data(), ng*ny, MPI_FLOAT, left, tag_x_r2l,
  //                 recv_column_left.data(), ng*ny, MPI_FLOAT, left, tag_x_l2r,
  //                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);  
  //   }
  //   if(right != MPI_PROC_NULL && ng > 0) {
  //     MPI_Sendrecv(send_column_right.data(), ng*ny, MPI_FLOAT, right, tag_x_l2r,
  //                 recv_column_right.data(), ng*ny, MPI_FLOAT, right, tag_x_r2l,
  //                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);  
  //   }
  //   if(up != MPI_PROC_NULL && ng > 0) {
  //     MPI_Sendrecv(send_row_top.data(), ng*nx, MPI_FLOAT, up, tag_y_b2t,
  //                 recv_row_top.data(), ng*nx, MPI_FLOAT, up, tag_y_t2b,
  //                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  //   }
  //   if(down != MPI_PROC_NULL && ng > 0) {
  //     MPI_Sendrecv(send_row_bottom.data(), ng*nx, MPI_FLOAT, down, tag_y_t2b,
  //                 recv_row_bottom.data(), ng*nx, MPI_FLOAT, down, tag_y_b2t,
  //                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  //   }

  //   // unpack received ghost layers into u
  //   for(int g=0; g < ng; ++g) {
  //     // Unpack left and right ghost layer
  //     for(int j=0; j < ny; ++j) {
  //       if(left != MPI_PROC_NULL) {
  //         U[index(g, ng+j)] = recv_column_left[g*ny + j]; // left ghost layer
  //       }
  //       if(right != MPI_PROC_NULL) {
  //         U[index(ng + nx + g, ng + j)] = recv_column_right[g*ny + j]; // right ghost layer 
  //       }
  //     }
  //     // Unpack top and bottom ghost layer
  //     for(int i=0; i < nx; ++i) {
  //       if(down != MPI_PROC_NULL) {
  //         U[index(ng + i, g)] = recv_row_bottom[g*nx + i]; // bottom ghost layer
  //       }
  //       if(up != MPI_PROC_NULL) {
  //         U[index(ng + i, ng + ny + g)] = recv_row_top[g*nx + i]; // top ghost layer   
  //       }
  //     }
  //   }
  // };



  // Example: Jacobi iteration
  const float omega = 1.0; // relaxation parameter
  const int max_iter = 200000;
  const float tolerance = 1e-6;

  float local_error = 0.0;
  bool converged = false;
  const float inv_hx2 = 1.0 / (hx * hx);
  const float inv_hy2 = 1.0 / (hy * hy);
  const float denom = 2.0 * (inv_hx2 + inv_hy2);

  for(int iter = 0; iter < max_iter; ++iter) {
    local_error = 0.0;
    halo_exchange.exchange(u); // Update ghost layers before computation
    for(int i = ng; i < nx + ng; ++i) {
      for(int j = ng; j < ny + ng; ++j) {
        
        auto [global_i, global_j] = local_to_global(i-ng, j-ng);
        if(global_i == 0 || global_i == Nx - 1 || global_j == 0 || global_j == Ny - 1) {
          u_new[index(i, j)] = 0.0; // Dirichlet boundary condition
        }
        else {
          float u_old = u[index(i, j)];
          float u_new_val = ((u[index(i-1, j)] + u[index(i+1, j)]) * inv_hx2 + \
                            (u[index(i, j-1)] + u[index(i, j+1)]) * inv_hy2 + f[index(i, j)]) / denom;
          u_new[index(i, j)] = (1.0 - omega) * u_old + omega * u_new_val;
        }
        local_error = std::max(local_error, std::abs(u_new[index(i, j)] - u[index(i, j)]));
      }
    }
    // Compute global error
    float global_error;
    MPI_Allreduce(&local_error, &global_error, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

    if(rank == 0 && iter % 1000 == 0) {
      printf("Iteration %d: Global error = %e\n", iter, global_error);
    }

    if (global_error < tolerance) {
      converged = true;
    }

    // Swap arrays
    std::swap(u, u_new);
    if(converged) break;
  }

  // Error analysis and output results
  float local_l2_error = 0.0;
  float local_linf_error = 0.0;
  for(int i = ng; i < nx + ng; ++i) {
    for(int j = ng; j < ny + ng; ++j) {
      auto [global_i, global_j] = local_to_global(i-ng, j-ng);
      float x = global_i * hx;
      float y = global_j * hy;
      float exact = std::sin(M_PI * x) * std::sin(M_PI * y);
      float error = std::abs(u[index(i, j)] - exact);
      local_l2_error += error * error;
      local_linf_error = std::max(local_linf_error, error);
    }
  }

  // Compute global errors
  float global_l2_error;
  float global_linf_error;
  MPI_Allreduce(&local_l2_error, &global_l2_error, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local_linf_error, &global_linf_error, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

  global_l2_error = std::sqrt(global_l2_error * hx * hy); // Normalize L2 error by total number of points

  if(rank == 0) {
    printf("Global L2 error = %e\n", global_l2_error);
    printf("Global L-infinity error = %e\n", global_linf_error);
  }

  MPI_Finalize();
  return 0;
}