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

struct Box {
    float h[9], h_inv[9];
    float thickness_x, thickness_y, thickness_z;
    bool pbc_x, pbc_y, pbc_z;
	__device__ float3 to_scaled(const float3& r) const {
	    float3 s;
	    s.x = h_inv[0] * r.x + h_inv[1] * r.y + h_inv[2] * r.z;
	    s.y = h_inv[3] * r.x + h_inv[4] * r.y + h_inv[5] * r.z;
	    s.z = h_inv[6] * r.x + h_inv[7] * r.y + h_inv[8] * r.z;
	    return s;
	}
	
	__device__ float3 to_cartesian(const float3& s) const {
	    float3 r;
	    r.x = h[0] * s.x + h[1] * s.y + h[2] * s.z;
	    r.y = h[3] * s.x + h[4] * s.y + h[5] * s.z;
	    r.z = h[6] * s.x + h[7] * s.y + h[8] * s.z;
	    return r;
	}
	
	__device__ float3 apply_mic(float3& dr) const {
	    float3 s = to_scaled(dr);
	    if (pbc_x) s.x -= nearbyintf(s.x);
	    if (pbc_y) s.y -= nearbyintf(s.y);
	    if (pbc_z) s.z -= nearbyintf(s.z);
	    float3 dr_ = to_cartesian(s);
	    return dr_;
	}
    float get_volume(void);
    void find_inv(void);
    void find_thickness(void);
    void update_bins(float rc, int& nx, int& ny, int& nz);
};
