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
#include "box.cuh"

static float get_area_one_direction(const float* a, const float* b)
{
  float s1 = a[1] * b[2] - a[2] * b[1];
  float s2 = a[2] * b[0] - a[0] * b[2];
  float s3 = a[0] * b[1] - a[1] * b[0];
  return sqrt(s1 * s1 + s2 * s2 + s3 * s3);
}

float Box::get_volume(void) {
    float volume;
    volume = abs(
        h[0] * (h[4] * h[8] - h[5] * h[7]) +
        h[1] * (h[5] * h[6] - h[3] * h[8]) +
        h[2] * (h[3] * h[7] - h[4] * h[6]));
    return volume;
}

void Box::find_inv(void) {
    float v_inv = 1.0f / get_volume();
    h_inv[0] = h[4] * h[8] - h[5] * h[7];
    h_inv[1] = h[2] * h[7] - h[1] * h[8];
    h_inv[2] = h[1] * h[5] - h[2] * h[4];
    h_inv[3] = h[5] * h[6] - h[3] * h[8];
    h_inv[4] = h[0] * h[8] - h[2] * h[6];
    h_inv[5] = h[2] * h[3] - h[0] * h[5];
    h_inv[6] = h[3] * h[7] - h[4] * h[6];
    h_inv[7] = h[1] * h[6] - h[0] * h[7];
    h_inv[8] = h[0] * h[4] - h[1] * h[3];
    for (int n = 9; n < 18; n++)  h[n] *= v_inv;
}


void Box::find_thickness(void) {
    float volume = get_volume();
    float a[3] = {h[0], h[3], h[6]};
    float b[3] = {h[1], h[4], h[7]};
    float c[3] = {h[2], h[5], h[8]};
    thickness_x = volume / get_area_one_direction(b, c);
    thickness_y = volume / get_area_one_direction(c, a);
    thickness_z = volume / get_area_one_direction(a, b);
}

void Box::update_bins(float rc, int& nx, int& ny, int& nz) {
    nx = floor(thickness_x / rc);
    ny = floor(thickness_y / rc);
    nz = floor(thickness_z / rc);
}
