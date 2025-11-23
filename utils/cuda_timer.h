#pragma once
#include <cstdio>

struct GpuTimer {
    cudaEvent_t start, stop;
    bool enabled;
    int repeat;
    int warmup;

    GpuTimer(int repeat_ = 1000, int warmup_ = 10, bool enable=true)
    {
#if ENABLE_KERNEL_PROFILE
    enabled = enable;
    repeat = repeat_;
    warmup = warmup_;
#else
    enabled = false;
    repeat = 1;
    warmup = 0;
#endif
        if (enabled) {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
        }
    }

    // 测量 kernel
    template<typename KernelFunc>
    void Measure(int version, KernelFunc kernel, dim3 grid, dim3 block, float* d_in, float* d_out) {
        if (!enabled) {
            kernel<<<grid, block>>>(d_in, d_out);
            cudaDeviceSynchronize();
            return;
        }

        // warm-up
        for (int i = 0; i < warmup; ++i)
            kernel<<<grid, block>>>(d_in, d_out);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < repeat; ++i)
            kernel<<<grid, block>>>(d_in, d_out);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        printf("Kernel reduce%d avg time: %.6f ms (warmup=%d, repeat=%d)\n",
            version, ms / repeat, warmup, repeat);
    }


    ~GpuTimer() {
        if (enabled) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    }
};