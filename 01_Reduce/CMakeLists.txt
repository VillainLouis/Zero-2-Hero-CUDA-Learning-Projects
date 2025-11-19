# 添加可执行文件
add_executable(reduce_v0_global_memory reduce_v0_global_memory.cu)

# 链接 CUDA 库：统一使用现代导入目标，移除旧版变量
target_link_libraries(reduce_v0_global_memory PRIVATE 
    CUDA::cudart 
    CUDA::cublas  # 替换 ${CUDA_cublas_LIBRARY}
)

# Debug 模式添加 CUDA 调试选项
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_v0_global_memory PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-G>        # 生成 CUDA 调试信息
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo> # 可选：生成行号信息，便于调试
    )
endif()

# 可选：指定 CUDA 架构（避免自动检测的兼容性问题）
set_target_properties(reduce_v0_global_memory PROPERTIES
    CUDA_ARCHITECTURES 80  # 根据你的 GPU 架构调整，如 75=Turing、80=Ampere、90=Hopper
)