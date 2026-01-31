/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/
/*
    Modified by Gegejun to combine with pybind11
*/

#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "box.cuh"
#include <vector>    // 为 std::vector 提供定义
#include <cstdint>   // 为 uintptr_t 提供定义

#define CHECK(call)                                                     \
  do {                                                                  \
    const cudaError_t error_code = (call);                              \
    if (error_code != cudaSuccess) {                                    \
      fprintf(stderr, "CUDA Error:\n");                                 \
      fprintf(stderr, "  File: %s\n", __FILE__);                        \
      fprintf(stderr, "  Line: %d\n", __LINE__);                        \
      fprintf(stderr, "  Code: %d\n", error_code);                      \
      fprintf(stderr, "  Text: %s\n", cudaGetErrorString(error_code));  \
      exit(1);                                                          \
    }                                                                   \
  } while (0)

struct CellIndex {
    int ix, iy, iz;
    int id;
};


__device__ __forceinline__ int wrap(int a, int n) {
    a = a % n;
    return a < 0 ? a + n : a;
};

static __device__ CellIndex find_cell_id(
    const Box& box,
    const float3& r,
    const float rc_inv,
    const int nx, const int ny, const int nz) 
{
    CellIndex idx;
    float3 s = box.to_scaled(r);
    idx.ix = wrap(static_cast<int>(floorf(s.x * box.thickness_x * rc_inv)), nx);
    idx.iy = wrap(static_cast<int>(floorf(s.y * box.thickness_y * rc_inv)), ny);
    idx.iz = wrap(static_cast<int>(floorf(s.z * box.thickness_z * rc_inv)), nz);
    idx.id = idx.ix + nx * (idx.iy + ny * idx.iz);
    return idx;
};


class NeighborList {
public:
    NeighborList(int _N, int _MN, float _rc);
    void update_box(std::vector<float> h, bool pbc_x, bool pbc_y, bool pbc_z);
    int find_neighbor(uintptr_t x_ptr, uintptr_t y_ptr, uintptr_t z_ptr);
    void find_cell_list(const float* d_x, const float* d_y, const float* d_z);
    void convert_ijS(uintptr_t idx_i_ptr, uintptr_t idx_j_ptr, uintptr_t offset_ptr);
    int N_neighbor;
    //float *d_offset;
    //int *d_idx_i, *d_idx_j;
private:
    Box box;
    const int N;
    const int MN;
    const float rc, rc_inv;
    int nx, ny, nz, N_cell;
    int *d_cell_contents;
    int *d_cell_count, *d_cell_count_sum;
    int *d_NN, *d_NN_sum;
    int *d_NL;
    float *d_Noffset;
};
