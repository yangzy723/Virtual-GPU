cmake_minimum_required(VERSION 3.18)

project(GPUScheduler LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(VGPU_BUILD_TESTS "Build GPU scheduler tests" ON)

# ── Daemon ───────────────────────────────────────────────────────────────────

add_executable(gpu_scheduler
    src/daemon/main.cpp
    src/daemon/scheduler.cpp
)
target_include_directories(gpu_scheduler PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(gpu_scheduler PRIVATE pthread)

# ── Shim library ─────────────────────────────────────────────────────────────
# Only libcuda.so is needed. The CUDA Runtime API (libcudart) internally calls
# the Driver API, so intercepting libcuda catches everything.

add_library(vgpu_cuda_driver_shim SHARED
    src/shim/intercept.cpp
)
set_target_properties(vgpu_cuda_driver_shim PROPERTIES OUTPUT_NAME cuda)
target_include_directories(vgpu_cuda_driver_shim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vgpu_cuda_driver_shim PRIVATE dl pthread)
target_compile_definitions(vgpu_cuda_driver_shim PRIVATE _GNU_SOURCE)

# ── Tests ────────────────────────────────────────────────────────────────────

if(VGPU_BUILD_TESTS)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_FOUND)
        add_custom_target(scheduler_test
            COMMAND ${Python3_EXECUTABLE} -m unittest -v
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vgpu.py
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            DEPENDS gpu_scheduler vgpu_cuda_driver_shim
            USES_TERMINAL
        )
    endif()
endif()
