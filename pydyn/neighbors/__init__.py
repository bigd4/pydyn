"""Neighbor list calculation modules for efficient force computation.

This module provides neighbor list implementations optimized for GPU computation
using CUDA, enabling efficient force calculations for large molecular systems.

邻域列表模块。

该模块提供了针对 GPU 计算优化的邻域列表实现，使用 CUDA 进行加速，
可以高效地计算大型分子系统的相互作用。
"""

from .cuda_neighbor import CudaNeighborList

__all__ = [
    "CudaNeighborList",
]
