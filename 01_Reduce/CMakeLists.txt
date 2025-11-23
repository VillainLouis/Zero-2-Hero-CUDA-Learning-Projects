# 1. 全局定义预处理器宏（放在 add_executable 之前）
add_definitions(-DENABLE_KERNEL_PROFILE=1)

# 添加可执行文件
add_executable(reduce reduce.cu ../utils/cuda_timer.h)

# 链接 CUDA 库：统一使用现代导入目标，移除旧版变量
target_link_libraries(reduce PRIVATE 
    CUDA::cudart 
    CUDA::cublas  # 替换 ${CUDA_cublas_LIBRARY}
)

# Debug 模式添加 CUDA 调试选项
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-G>        # 生成 CUDA 调试信息
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo> # 可选：生成行号信息，便于调试
    )
endif()

# 可选：指定 CUDA 架构（避免自动检测的兼容性问题）
set_target_properties(reduce PROPERTIES
    CUDA_ARCHITECTURES 86  # 根据你的 GPU 架构调整，如 75=Turing、80=Ampere、90=Hopper | RTX 3090 / 3080 / 3070 Ampere (GA102/GA104) 8.6（即 86）
)