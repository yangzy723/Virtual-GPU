cmake_minimum_required(VERSION 3.18)

project(VirtualGPU LANGUAGES C CXX)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(VGPU_BUILD_TESTS "Build Virtual-GPU smoke tests" ON)

set(VGPU_COMMON_SOURCES
    src/common/context_registry.cpp
    src/common/fatbin_parser.cpp
    src/common/kernel_registry.cpp
    src/common/rpc_client.cpp
    src/common/dlopen_hook.cpp
)

function(vgpu_configure_shared_target target)
    target_include_directories(${target}
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(${target}
        PRIVATE
            dl
            pthread
    )
    target_compile_definitions(${target} PRIVATE _GNU_SOURCE)
endfunction()

function(vgpu_add_cuda_test target source)
    add_executable(${target} ${source})
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set_target_properties(${target} PROPERTIES
        CUDA_STANDARD 14
        CUDA_STANDARD_REQUIRED ON
        CUDA_RUNTIME_LIBRARY Shared
    )
endfunction()

add_library(vgpu_intercept SHARED
    src/frontend/interceptor.cpp
    ${VGPU_COMMON_SOURCES}
)
vgpu_configure_shared_target(vgpu_intercept)

add_library(vgpu_cudart_shim SHARED
    src/frontend/interceptor.cpp
    ${VGPU_COMMON_SOURCES}
)
set_target_properties(vgpu_cudart_shim PROPERTIES OUTPUT_NAME cudart)
vgpu_configure_shared_target(vgpu_cudart_shim)

add_library(vgpu_cuda_driver_shim SHARED
    src/frontend/driver_interceptor.cpp
    ${VGPU_COMMON_SOURCES}
)
set_target_properties(vgpu_cuda_driver_shim PROPERTIES OUTPUT_NAME cuda)
vgpu_configure_shared_target(vgpu_cuda_driver_shim)

# Convenience library for PyTorch integration
# Combines dlopen hooking and preload initialization
add_library(vgpu_preload_init SHARED
    src/common/cuda_preload_init.cpp
    src/common/dlopen_hook.cpp
)
set_target_properties(vgpu_preload_init PROPERTIES OUTPUT_NAME vgpu_preload_init)
target_include_directories(vgpu_preload_init PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vgpu_preload_init PRIVATE dl pthread)
target_compile_definitions(vgpu_preload_init PRIVATE _GNU_SOURCE)

add_executable(vgpu_server
    src/backend/server_main.cpp
    src/backend/cuda_driver_loader.cpp
    src/backend/cuda_runtime_loader.cpp
    src/common/fatbin_parser.cpp
)
target_include_directories(vgpu_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vgpu_server PRIVATE dl pthread)

if(VGPU_BUILD_TESTS)
    add_executable(vgpu_runtime_shim_smoke_test test/runtime_shim_smoke_test.cpp)
    target_include_directories(vgpu_runtime_shim_smoke_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(vgpu_runtime_shim_smoke_test PRIVATE vgpu_cudart_shim)

    if(CMAKE_CUDA_COMPILER)
        vgpu_add_cuda_test(vgpu_runtime_memory_smoke_test test/runtime_memory_smoke_test.cu)
        vgpu_add_cuda_test(vgpu_runtime_async_event_test test/runtime_async_event_test.cu)
        vgpu_add_cuda_test(vgpu_runtime_kernel_launch_test test/runtime_kernel_launch_test.cu)
    endif()
endif()
