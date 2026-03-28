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



#include "neighbor.cuh"
#include "box.cuh"
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <cstring>


static __global__ void find_cell_counts(
  const Box box,
  const int N,
  int* cell_count,
  const float* x,
  const float* y,
  const float* z,
  const int nx,
  const int ny,
  const int nz,
  const float rc_inv)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    float3 r = {x[n1], y[n1], z[n1]};
    CellIndex idx = find_cell_id(box, r, rc_inv, nx, ny, nz);
    atomicAdd(&cell_count[idx.id], 1);
  }
}

static __global__ void find_cell_contents(
  const Box box,
  const int N,
  int* cell_count,
  const int* cell_count_sum,
  int* cell_contents,
  const float* x,
  const float* y,
  const float* z,
  const int nx,
  const int ny,
  const int nz,
  const float rc_inv)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    float3 r = {x[n1], y[n1], z[n1]};
    CellIndex idx = find_cell_id(box, r, rc_inv, nx, ny, nz);
    const int ind = atomicAdd(&cell_count[idx.id], 1);
    cell_contents[cell_count_sum[idx.id] + ind] = n1;
  }
}


__global__ void find_ijS(
    const int N,
    const int MN,
    int* NN,
    const int* NN_sum,
    int* NL,
    float* Noffset,
    int* idx_i, 
    int* idx_j,
    float* offset)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (n1 < N) {
        int NN_start = NN_sum[n1];
        int offset_base = n1 * MN;
        for (int idx = 0; idx < NN[n1]; idx++) {
            // 更新idx_i、idx_j、offset值
            idx_i[NN_start + idx] = n1;
            idx_j[NN_start + idx] = NL[offset_base + idx];
            // 批量写入offset数组
            int base_idx = (NN_start + idx) * 3;
            int offset_idx = (offset_base + idx) * 3;
            offset[base_idx]     = Noffset[offset_idx];
            offset[base_idx + 1] = Noffset[offset_idx + 1];
            offset[base_idx + 2] = Noffset[offset_idx + 2];
        }
    }
}

__global__ void gpu_find_neighbor_ON1(
    const Box box,
    const int N,
    const int* __restrict__ cell_counts,
    const int* __restrict__ cell_count_sum,
    const int* __restrict__ cell_contents,
    int* __restrict__ NN,
    int* __restrict__ NL,
    float* __restrict__ Noffset,
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ z,
    const int nx,
    const int ny,
    const int nz,
    const float rc_inv,
    const float cutoff_square,
    const int MN)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (n1 >= N) return;

    float3 r1 = {x[n1], y[n1], z[n1]};
    CellIndex idx = find_cell_id(box, r1, rc_inv, nx, ny, nz);
    int count = 0;

    int xl = box.pbc_x ? 1 : 0;
    int yl = box.pbc_y ? 1 : 0;
    int zl = box.pbc_z ? 1 : 0;

    for (int k = -zl; k <= zl; ++k) {
        for (int j = -yl; j <= yl; ++j) {
            for (int i = -xl; i <= xl; ++i) {
                int ix = wrap(idx.ix + i, nx);
                int iy = wrap(idx.iy + j, ny);
                int iz = wrap(idx.iz + k, nz);
                int cell = ix + nx * (iy + ny * iz);

                int start = cell_count_sum[cell];
                int ncell = cell_counts[cell];

                for (int m = 0; m < ncell; ++m) {
                    int n2 = cell_contents[start + m];
                    if (n2 == n1 || n2 >= N) continue;
                    float3 dr = {x[n2] - r1.x, y[n2] - r1.y, z[n2] - r1.z};
                    float3 dr_ = box.apply_mic(dr);
                    float d2 = dr_.x * dr_.x + dr_.y * dr_.y + dr_.z * dr_.z;
                    if (d2 < cutoff_square && count < MN) {
                        NL[n1 * MN + count] = n2;
                        Noffset[(n1 * MN + count) * 3] = dr_.x - dr.x;
                        Noffset[(n1 * MN + count) * 3 + 1] = dr_.y - dr.y;
                        Noffset[(n1 * MN + count) * 3 + 2] = dr_.z - dr.z;
                        count++;
                    }
                }
            }
        }
    }
    NN[n1] = count;
}


NeighborList::NeighborList(int _N, int _MN, float _rc)
    : N(_N), MN(_MN), rc(_rc), rc_inv(1.0f / _rc)  {
    box = {};
    CHECK(cudaMalloc((void**)&d_cell_contents, N * sizeof(int)));
    CHECK(cudaMalloc((void**)&d_NN, N * sizeof(int)));
    CHECK(cudaMalloc((void**)&d_NN_sum, N * sizeof(int)));
    CHECK(cudaMalloc((void**)&d_NL, N * MN * sizeof(int)));
    CHECK(cudaMalloc((void**)&d_Noffset, N * MN * 3 * sizeof(float)));
    N_neighbor = 0;
    //cudaMalloc((void**)&d_idx_i, N * MN * sizeof(int));
    //cudaMalloc((void**)&d_idx_j, N * MN * sizeof(int));
    //cudaMalloc((void**)&d_offset, N * MN * 3 * sizeof(float));
    N_cell = 0;
    CHECK(cudaMalloc(&d_cell_count, N_cell * sizeof(int)));
    CHECK(cudaMalloc(&d_cell_count_sum, N_cell * sizeof(int)));
}

NeighborList::~NeighborList() {
    cudaFree(d_cell_contents);
    cudaFree(d_NN);
    cudaFree(d_NN_sum);
    cudaFree(d_NL);
    cudaFree(d_Noffset);
    cudaFree(d_cell_count);
    cudaFree(d_cell_count_sum);
}

void NeighborList::update_box(std::vector<float> h, bool pbc_x, bool pbc_y, bool pbc_z) {
    for (int i = 0; i < 9; ++i) box.h[i] = h[i];
    box.pbc_x = pbc_x;
    box.pbc_y = pbc_y;
    box.pbc_z = pbc_z;

    box.find_inv();
    box.find_thickness();
    box.update_bins(rc, nx, ny, nz);
    // If the number of cells (nx * ny * nz) has changed, we need to reallocate memory
    if (nx * ny * nz > N_cell) {
        // Free old memory before reallocation
        cudaFree(d_cell_count);
        cudaFree(d_cell_count_sum);
        // Update N_cell and allocate new memory
        N_cell = nx * ny * nz;
        cudaMalloc(&d_cell_count, N_cell * sizeof(int));
        cudaMalloc(&d_cell_count_sum, N_cell * sizeof(int));
    }
}


void NeighborList::find_cell_list(const float* d_x, const float* d_y, const float* d_z) {
    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;
    const float rc_inv = 1.0f / rc;

    // 清零
    CHECK(cudaMemset(d_cell_count, 0, sizeof(int) * N_cell));
    CHECK(cudaMemset(d_cell_count_sum, 0, sizeof(int) * N_cell));
    CHECK(cudaMemset(d_cell_contents, 0, sizeof(int) * N));

    // 统计每个 cell 原子数
    find_cell_counts<<<grid_size, block_size>>>(
        box, N, d_cell_count, d_x, d_y, d_z, nx, ny, nz, rc_inv);
    CHECK(cudaGetLastError());

    // exclusive_scan 得到 cell_count_sum
    thrust::exclusive_scan(
        thrust::device,
        d_cell_count,
        d_cell_count + N_cell,
        d_cell_count_sum);

    // 重置 cell_count 用于 find_cell_contents 的 atomicAdd
    CHECK(cudaMemset(d_cell_count, 0, sizeof(int) * N_cell));

    // 填充 cell_contents
    find_cell_contents<<<grid_size, block_size>>>(
        box, N,
        d_cell_count,
        d_cell_count_sum,
        d_cell_contents,
        d_x, d_y, d_z,
        nx, ny, nz,
        rc_inv);
    CHECK(cudaGetLastError());
}


void NeighborList::convert_ijS(uintptr_t idx_i_ptr, uintptr_t idx_j_ptr, uintptr_t offset_ptr){
    const int block_size = 256;
    const int grid_size  = (N + block_size - 1) / block_size;
    int* d_idx_i          = reinterpret_cast<int*>(idx_i_ptr);
    int* d_idx_j          = reinterpret_cast<int*>(idx_j_ptr);
    float* d_offset       = reinterpret_cast<float*>(offset_ptr);

    thrust::exclusive_scan(
        thrust::device,
        d_NN,
        d_NN + N,
        d_NN_sum);

    find_ijS<<<grid_size, block_size>>>(
        N,
        MN,
        d_NN,
        d_NN_sum,
        d_NL,
        d_Noffset,
        d_idx_i,
        d_idx_j,
        d_offset);

    CHECK(cudaGetLastError());
}


int NeighborList::find_neighbor(uintptr_t x_ptr, uintptr_t y_ptr, uintptr_t z_ptr) {
    const int block_size = 256;
    const int grid_size  = (N + block_size - 1) / block_size;
    const float* d_x = reinterpret_cast<float*>(x_ptr);
    const float* d_y = reinterpret_cast<float*>(y_ptr);
    const float* d_z = reinterpret_cast<float*>(z_ptr);
    find_cell_list(d_x, d_y, d_z);
    CHECK(cudaGetLastError());
    // 清零
    CHECK(cudaMemset(d_NN, 0, sizeof(int) * N));
    CHECK(cudaMemset(d_NN_sum, 0, sizeof(int) * N));
    CHECK(cudaMemset(d_NL, 0, sizeof(int) * N * MN));
    gpu_find_neighbor_ON1<<<grid_size, block_size>>>(
        box,
        N,
        d_cell_count,
        d_cell_count_sum,
        d_cell_contents,
        d_NN, d_NL, d_Noffset,
        d_x, d_y, d_z,
        nx, ny, nz,
        1.0f / rc,
        rc * rc,
        MN);
    CHECK(cudaGetLastError());
    N_neighbor = thrust::reduce(thrust::device, d_NN, d_NN + N, 0, thrust::plus<int>());
    return N_neighbor;
    //return 0;
}
