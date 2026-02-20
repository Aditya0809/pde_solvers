#pragma once
#include <mpi.h>
#include <iostream>
#include "decomp2d.hpp"
#include <vector>



class HaloExchange
{
    MPI_Comm comm_;
    int rank_, size_;
    int left_, right_, up_, down_;
    int nghost_;
    int local_nx_, local_ny_; // local grid size without ghost cells
    //int i0_, j0_; // global index of the first local grid point (excluding ghost cells)
    //int global_nx_, global_ny_; // global grid size
    std::vector<float> send_column_left, send_column_right, recv_column_left, recv_column_right;
    std::vector<float> send_row_top, send_row_bottom, recv_row_top, recv_row_bottom;
    const int tag_x_l2r = 10; // tag for left to right communication
    const int tag_x_r2l = 11; // tag for right to left communication
    const int tag_y_b2t = 12; // tag for bottom to top communication
    const int tag_y_t2b = 13; // tag for top to bottom communication

public:
    HaloExchange(const Decomp2D &decomp){
        comm_ = decomp.comm();
        rank_ = decomp.rank();
        size_ = decomp.size();
        left_ = decomp.left();
        right_ = decomp.right();
        up_ = decomp.up();
        down_ = decomp.down();
        nghost_ = decomp.nghost();
        local_nx_ = decomp.nx();
        local_ny_ = decomp.ny();
        // i0_ = decomp.i0();
        // j0_ = decomp.j0();
        // global_nx_ = decomp.Nx();
        // global_ny_ = decomp.Ny();
        send_column_left.resize(nghost_ * local_ny_);
        recv_column_left.resize(nghost_ * local_ny_);
        send_column_right.resize(nghost_ * local_ny_);
        recv_column_right.resize(nghost_ * local_ny_);
        send_row_top.resize(nghost_ * local_nx_);
        recv_row_top.resize(nghost_ * local_nx_);
        send_row_bottom.resize(nghost_ * local_nx_);
        recv_row_bottom.resize(nghost_ * local_nx_);
    }

    void exchange(std::vector<float> &U) {
        int nx_tot = local_nx_ + 2*nghost_; // total local grid size including ghost cells
        int ny_tot = local_ny_ + 2*nghost_; // total local grid size including ghost cells

        // assert that U has the correct size
        if (static_cast<int>(U.size()) != nx_tot * ny_tot) {
            std::cerr << "Error: U has incorrect size. Expected " << nx_tot * ny_tot << " but got " << U.size() << std::endl;
            MPI_Abort(comm_, 1);
        }

        int stride = ny_tot; // Assuming row-major order
        // Prepare send buffers
        for(int g=0; g < nghost_; ++g) {
            // Prepare left and rigtht ghost layer to send
            for(int j=0; j < local_ny_; ++j) {
                send_column_left[g*local_ny_ + j] = U[(nghost_+ g)*stride + nghost_ + j]; // left ghost layer
                send_column_right[g*local_ny_ + j] = U[(nghost_ + local_nx_ - nghost_ + g)*stride + nghost_ + j]; // right ghost layer
            }
            // Prepare top and bottom ghost layer to send
            for(int i=0; i < local_nx_; ++i) {
                send_row_bottom[g*local_nx_ + i] = U[(nghost_ + i)*stride + nghost_ + g]; // bottom ghost layer
                send_row_top[g*local_nx_ + i] = U[(nghost_ + i)*stride + nghost_ + local_ny_ - nghost_ + g]; // top ghost layer
            }
        }

        // Perform communication with neighbors using MPI_Sendrecv
        if(left_ != MPI_PROC_NULL && nghost_ > 0) {
            MPI_Sendrecv(send_column_left.data(), nghost_*local_ny_, MPI_FLOAT, left_, tag_x_r2l,
                        recv_column_left.data(), nghost_*local_ny_, MPI_FLOAT, left_, tag_x_l2r,
                        comm_, MPI_STATUS_IGNORE);  
        }
        if(right_ != MPI_PROC_NULL && nghost_ > 0) {
            MPI_Sendrecv(send_column_right.data(), nghost_*local_ny_, MPI_FLOAT, right_, tag_x_l2r,
                        recv_column_right.data(), nghost_*local_ny_, MPI_FLOAT, right_, tag_x_r2l,
                        comm_, MPI_STATUS_IGNORE);  
        }
        if(up_ != MPI_PROC_NULL && nghost_ > 0) {
            MPI_Sendrecv(send_row_top.data(), nghost_*local_nx_, MPI_FLOAT, up_, tag_y_b2t,
                        recv_row_top.data(), nghost_*local_nx_, MPI_FLOAT, up_, tag_y_t2b,
                        comm_, MPI_STATUS_IGNORE);
        }
        if(down_ != MPI_PROC_NULL && nghost_ > 0) {
            MPI_Sendrecv(send_row_bottom.data(), nghost_*local_nx_, MPI_FLOAT, down_, tag_y_t2b,
                        recv_row_bottom.data(), nghost_*local_nx_, MPI_FLOAT, down_, tag_y_b2t,
                        comm_, MPI_STATUS_IGNORE);
        }

        // Unpack received ghost layers into U
        for(int g=0; g < nghost_; ++g) {
            // Unpack left and right ghost layer
            for(int j=0; j < local_ny_; ++j) {
                if(left_ != MPI_PROC_NULL) {
                    U[g*stride + nghost_ + j] = recv_column_left[g*local_ny_ + j]; // left ghost layer
                }
                if(right_ != MPI_PROC_NULL) {
                    U[(nghost_ + local_nx_ + g)*stride + nghost_ + j] = recv_column_right[g*local_ny_ + j]; // right ghost layer 
                }

            }
            // Unpack top and bottom ghost layer
            for(int i=0; i < local_nx_; ++i) {
                if(up_ != MPI_PROC_NULL) {
                    U[(nghost_ + i)*stride + nghost_ + local_ny_ + g] = recv_row_top[i + g*local_nx_]; // top ghost layer
                }
                if(down_ != MPI_PROC_NULL) {
                    U[(nghost_ + i)*stride +  g] = recv_row_bottom[i + g*local_nx_]; // bottom ghost layer
                }
            }
        }
    }

    



};
