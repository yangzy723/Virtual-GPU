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
    src/shim/daemon_channel.cpp
    src/shim/local_scheduler.cpp
    src/shim/scheduler_backend.cpp
)
set_target_properties(vgpu_cuda_driver_shim PROPERTIES OUTPUT_NAME cuda)
target_include_directories(vgpu_cuda_driver_shim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vgpu_cuda_driver_shim PRIVATE dl pthread)
target_compile_definitions(vgpu_cuda_driver_shim PRIVATE _GNU_SOURCE)

# ── Tests ────────────────────────────────────────────────────────────────────

if(VGPU_BUILD_TESTS)
    enable_testing()

    add_executable(local_scheduler_test
        tests/local_scheduler_test.cpp
        src/shim/local_scheduler.cpp
    )
    target_include_directories(local_scheduler_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/shim
    )
    target_link_libraries(local_scheduler_test PRIVATE pthread)
    add_test(NAME local_scheduler_test COMMAND local_scheduler_test)

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_FOUND)
        add_custom_target(scheduler_test
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vgpu.py -v
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            DEPENDS gpu_scheduler vgpu_cuda_driver_shim local_scheduler_test
            USES_TERMINAL
        )
    endif()
endif()
