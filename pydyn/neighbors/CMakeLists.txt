cmake_minimum_required(VERSION 3.20)
project(pydyn_neighbors LANGUAGES CXX CUDA)

# ------------------------------
# 1. 自动检测 Python
# ------------------------------
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
add_subdirectory(../../pybind11 pybind11_binary)

# ------------------------------
# 2. 自动检测 CUDA Toolkit
# ------------------------------
find_package(CUDAToolkit REQUIRED)

message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA toolkit include: ${CUDAToolkit_INCLUDE_DIRS}")

# ------------------------------
# 3. 自动检测 GPU 架构（sm80/sm90/…）
# ------------------------------
if (NOT CUDA_ARCHES)
    message(WARNING "CUDA_ARCHES missing, auto detecting...")
    find_program(NVIDIA_SMI nvidia-smi)

    if (NVIDIA_SMI)
        execute_process(
            COMMAND ${NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE GPU_CC
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        if (GPU_CC)
            string(REPLACE "." "" GPU_ARCH "${GPU_CC}")
            set(CUDA_ARCHES "${GPU_ARCH}")
            message(STATUS "Detected actual GPU arch: sm_${GPU_ARCH}")
        else()
            message(WARNING "Failed to detect compute capability, defaulting to sm_80")
            set(CUDA_ARCHES 80)
        endif()

    else()
        message(WARNING "nvidia-smi not found; defaulting to sm_80")
        set(CUDA_ARCHES 80)
    endif()
endif()
set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCHES}")
# ------------------------------
# 4. 编译CUDA部分
# ------------------------------
add_library(neighbor_cuda STATIC
    neighbor.cu
    box.cu
)

set_target_properties(neighbor_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(neighbor_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(neighbor_cuda
    PUBLIC CUDA::cudart
)


# ------------------------------
# 5. Build Python extension
# ------------------------------
pybind11_add_module(libneighbor binding.cpp)

target_link_libraries(libneighbor PRIVATE
    neighbor_cuda
    CUDA::cudart
)

target_include_directories(libneighbor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# ------------------------------
# 6. 放到 neighbors 路径下
# ------------------------------
set_target_properties(libneighbor PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
