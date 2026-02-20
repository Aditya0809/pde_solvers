#pragma once
#include <mpi.h>
#include <iostream>


class Decomp2D
{
    MPI_Comm comm_;
    int Nx_, Ny_; // global grid size
    int Px_, Py_; // process grid size
    int rank_, size_;
    int px_, py_; // process coordinates in the process grid
    int nx_, ny_; // local grid size for this process (without ghost cells)
    int i0_, i1_, j0_, j1_; // bounds are half open: [i0, i1), [j0, j1)
    int left_, right_, up_, down_;
    int nghost_; // number of ghost cells for communication

public:
    Decomp2D(MPI_Comm comm, int Nx, int Ny, int Px, int Py, int nghost = 0) : comm_(comm), Nx_(Nx), Ny_(Ny), Px_(Px), Py_(Py), nghost_(nghost)
    {
        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &size_);

        // check that the number of processes matches the decomposition
        if (size_ != Px_ * Py_)
        {
            if (rank_ == 0) {
                std::cerr << "Error: Number of processes must be equal to Px * Py" << std::endl;
            }
            MPI_Abort(comm_, 1);
        }

        // check if Nx, Ny, Px, Py are greater than 0
        if (Nx_ <= 0 || Ny_ <= 0 || Px_ <= 0 || Py_ <= 0)
        {
            if (rank_ == 0) {
                std::cerr << "Error: Nx, Ny, Px, Py must be greater than 0" << std::endl;
            }
            MPI_Abort(comm_, 1);
        }

        // Determine the process grid coordinates
        px_ = rank_ % Px_;
        py_ = rank_ / Px_;

        // Compute local grid size (handle cases where Nx or Ny is not divisible by Px or Py)
        int base_x = Nx_ / Px_;
        int base_y = Ny_ / Py_;
        int rem_x = Nx_ % Px_;
        int rem_y = Ny_ % Py_;

        if (px_ <rem_x) {
            nx_ = base_x + 1; // add one more column to the first (Nx % Px) processes in x direction
            i0_ = px_ * (base_x + 1);
        }
        else {
            nx_ = base_x;
            i0_ = (rem_x * (base_x + 1)) + (px_ - rem_x) * base_x;
        }
        i1_ = i0_ + nx_;

        if (py_ < rem_y) {
            ny_ = base_y + 1; // add one more row to the first (Ny % Py) processes in y direction
            j0_ = py_ * (base_y + 1);
        }
        else {
            ny_ = base_y;
            j0_ = (rem_y * (base_y + 1)) + (py_ - rem_y) * base_y;
        }
        j1_ = j0_ + ny_;

        // Check domain bounds
        if (i1_ < i0_ || i0_ < 0 || i1_ > Nx_ || j1_ < j0_ || j0_ < 0 || j1_ > Ny_)
        {
            std::cerr << "Error: Invalid local grid bounds for process " << rank_ << std::endl;
            MPI_Abort(comm_, 1);
        }


        // Determine neighbors (up, down, left, right)
        // Non-periodic boundaries: if there is no neighbor, set to MPI_PROC_NULL, add later support for periodic boundaries
        if (px_ > 0) left_ = rank_ - 1; // left
        else left_ = MPI_PROC_NULL; // no left neighbor
        if (px_ < Px_ - 1) right_ = rank_ + 1; // right
        else right_ = MPI_PROC_NULL; // no right neighbor
        if (py_ > 0) down_ = rank_ - Px_; // down
        else down_ = MPI_PROC_NULL; // no down neighbor
        if (py_ < Py_ - 1) up_ = rank_ + Px_; // up
        else up_ = MPI_PROC_NULL; // no up neighbor

        // Check nghost is non-negative
        if (nghost_ < 0) {
            if (rank_ == 0) {
                std::cerr << "Error: nghost must be non-negative" << std::endl;
            }
            MPI_Abort(comm_, 1);
        }

    }

    // Getters for local grid bounds
    int i0() const { return i0_; }
    int i1() const { return i1_; }
    int j0() const { return j0_; }
    int j1() const { return j1_; }
    
    // Getters for neighbors
    int left() const { return left_; }
    int right() const { return right_; }
    int up() const { return up_; }
    int down() const { return down_; }

    // Getter for px, py, Px, Py
    int px() const { return px_; }
    int py() const { return py_; }
    int Px() const { return Px_; }
    int Py() const { return Py_; }

    // Getter for local and global grid sizes
    int Nx() const { return Nx_; }
    int Ny() const { return Ny_; }
    int nx() const { return nx_; }
    int ny() const { return ny_; }

    // Getter for rank and size    
    int size() const { return size_; } 
    int rank() const { return rank_; }
    int nghost() const { return nghost_; }

    // Getter got comm
    MPI_Comm comm() const { return comm_; }

};
