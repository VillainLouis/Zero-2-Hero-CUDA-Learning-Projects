#include <bits/stdc++.h>
#include <cstdio>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>
#include "../utils/cuda_timer.h"

#define THREAD_PER_BLOCK 256

// v5: completely unrolling the for loop
template<unsigned int blockSize>
__device__ void warpReduce(volatile float *cache, int tid) { 
    // 'volatile' tells the compiler not to cache or reorder accesses to this memory,
    // ensuring that each thread reads the latest value from shared memory.
    if (blockSize >= 64) cache[tid] += cache[tid + 32];
    if (blockSize >= 32) cache[tid] += cache[tid + 16];
    if (blockSize >= 16) cache[tid] += cache[tid + 8];
    if (blockSize >= 8) cache[tid] += cache[tid + 4];
    if (blockSize >= 4) cache[tid] += cache[tid + 2];
    if (blockSize >= 2) cache[tid] += cache[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce5(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    // each thread loads two numbers from global and then write the additon of the two numbers to shared mem
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x; // each block cover 2*blockDim.x numbers
    sdata[tid] = d_in[i] + d_in[i+blockDim.x]; // thread tid handles d_in[i] and d_in[i + blockDim.x]
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512) {if (threadIdx.x < 256) sdata[tid] += sdata[tid+256]; __syncthreads();}
    if (blockSize >= 256) {if (threadIdx.x < 128) sdata[tid] += sdata[tid+128]; __syncthreads();}
    if (blockSize >= 128) {if (threadIdx.x < 64) sdata[tid] += sdata[tid+64]; __syncthreads();}

    // write result for this block to global mem
    if (tid < 32) {
        warpReduce<blockSize>(sdata, tid);
    }
    if (tid == 0) {
        d_out[blockIdx.x] = sdata[tid];
    }
}


// v4: unrolling last warp
__device__ void warpReduce(volatile float *cache, int tid) { 
    // 'volatile' tells the compiler not to cache or reorder accesses to this memory,
    // ensuring that each thread reads the latest value from shared memory.
    cache[tid] += cache[tid + 32];
    cache[tid] += cache[tid + 16];
    cache[tid] += cache[tid + 8];
    cache[tid] += cache[tid + 4];
    cache[tid] += cache[tid + 2];
    cache[tid] += cache[tid + 1];
}

__global__ void reduce4(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    // each thread loads two numbers from global and then write the additon of the two numbers to shared mem
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x; // each block cover 2*blockDim.x numbers
    sdata[tid] = d_in[i] + d_in[i+blockDim.x]; // thread tid handles d_in[i] and d_in[i + blockDim.x]
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 32 ; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid < 32) {
        warpReduce(sdata, tid);
    }
    if (tid == 0) {
        d_out[blockIdx.x] = sdata[tid];
    }
}

// v3: fix idle threads
__global__ void reduce3(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    // each thread loads two numbers from global and then write the additon of the two numbers to shared mem
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x; // each block cover 2*blockDim.x numbers
    sdata[tid] = d_in[i] + d_in[i+blockDim.x]; // thread tid handles d_in[i] and d_in[i + blockDim.x]
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0 ; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) {
        d_out[blockIdx.x] = sdata[tid];
    }
}


// v2: fix bank confilct
__global__ void reduce2(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    // each thread loads one element from global to shared mem
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = d_in[i];
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0 ; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) {
        d_out[blockIdx.x] = sdata[tid];
    }
}

// v1: fix warp divergence (TODO: bank conflict)
__global__ void reduce1(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    //each thread loads one element from global memory to shared mem
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int tid=threadIdx.x;
    sdata[tid]=d_in[i];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s = 1; s < blockDim.x; s *= 2){
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index+s];
        }
        __syncthreads();
    }
    
    // write result for this block to global mem
    if(tid==0)d_out[blockIdx.x]=sdata[tid];
}

// v0: Compute-bounded (warp-divergence)
__global__ void reduce0(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    // each thread loads one element from global to shared mem
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = d_in[i];
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (s * 2) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) {
        d_out[blockIdx.x] = sdata[tid];
    }
}

bool check(float *out,float *res,int n){
    for(int i=0;i<n;i++){
        if(out[i]!=res[i])
            return false;
    }
    return true;
}

int main(){
    const int N = 32*1024*1024;
    float *a=(float *)malloc(N*sizeof(float));
    float *d_a;
    cudaMalloc((void **)&d_a,N*sizeof(float));

    int block_num = N/THREAD_PER_BLOCK;
    float *out = (float *)malloc((N/THREAD_PER_BLOCK)*sizeof(float));
    float *d_out;
    cudaMalloc((void **)&d_out,(N/THREAD_PER_BLOCK)*sizeof(float));
    float *res = (float *)malloc((N/THREAD_PER_BLOCK)*sizeof(float));

    for(int i = 0;i < N; i++){
        a[i] = 1;
    }

    for(int i = 0; i < block_num; i++){
        float cur = 0;
        for(int j = 0;j < THREAD_PER_BLOCK; j++){
            cur += a[i*THREAD_PER_BLOCK+j];
        }
        res[i] = cur;
    }

    cudaMemcpy(d_a,a,N*sizeof(float),cudaMemcpyHostToDevice);

    int version = 0;

    printf("Enter the test kernel version (current: 0-5):");
    scanf("%d", &version);

    int repeat = 1000;
    int warmup = 10;
    GpuTimer timer(repeat, warmup, true);

    if (version == 0) {
        dim3 Grid( N/THREAD_PER_BLOCK,1);
        dim3 Block( THREAD_PER_BLOCK,1);
        timer.Measure(version, reduce0, Grid, Block, d_a, d_out);
    }
    else if (version == 1) {
        dim3 Grid( N/THREAD_PER_BLOCK,1);
        dim3 Block( THREAD_PER_BLOCK,1);
        timer.Measure(version, reduce1, Grid, Block, d_a, d_out);
        cudaMemcpy(out,d_out,block_num*sizeof(float),cudaMemcpyDeviceToHost);

        if(check(out,res,block_num))printf("the ans is right\n");
        else{
            printf("the ans is wrong\n");
            for(int i=0;i<block_num;i++){
                printf("%lf ",out[i]);
            }
            printf("\n");
        }
    }
    else if (version == 2) {
        dim3 Grid( N/THREAD_PER_BLOCK,1);
        dim3 Block( THREAD_PER_BLOCK,1);
        timer.Measure(version, reduce2, Grid, Block, d_a, d_out);
        
        cudaMemcpy(out,d_out,block_num*sizeof(float),cudaMemcpyDeviceToHost);

        if(check(out,res,block_num))printf("the ans is right\n");
        else{
            printf("the ans is wrong\n");
            for(int i=0;i<block_num;i++){
                printf("%lf ",out[i]);
            }
            printf("\n");
        }
    }
    else if (version == 3) {
        int NUM_PER_BLOCK = 2*THREAD_PER_BLOCK; // each thread load and add 2 data elements 
        int BLOCK_NUM = N / NUM_PER_BLOCK;
        for(int i = 0; i < BLOCK_NUM; i++){
            float cur = 0;
            for(int j = 0;j < NUM_PER_BLOCK; j++){
                cur += a[i*NUM_PER_BLOCK+j];
            }
            res[i] = cur;
        }
        dim3 Grid( BLOCK_NUM, 1);
        dim3 Block( THREAD_PER_BLOCK, 1);
        timer.Measure(version, reduce3, Grid, Block, d_a, d_out);
        cudaMemcpy(out,d_out,BLOCK_NUM*sizeof(float),cudaMemcpyDeviceToHost);

        if(check(out,res,BLOCK_NUM))printf("the ans is right\n");
        else{
            printf("the ans is wrong\n");
            for(int i=0;i<BLOCK_NUM;i++){
                printf("%lf ",out[i]);
            }
            printf("\n");
        }
    }
    else if (version == 4) {
        int NUM_PER_BLOCK = 2*THREAD_PER_BLOCK; // each thread load and add 2 data elements 
        int BLOCK_NUM = N / NUM_PER_BLOCK;
        for(int i = 0; i < BLOCK_NUM; i++){
            float cur = 0;
            for(int j = 0;j < NUM_PER_BLOCK; j++){
                cur += a[i*NUM_PER_BLOCK+j];
            }
            res[i] = cur;
        }
        dim3 Grid( BLOCK_NUM, 1);
        dim3 Block( THREAD_PER_BLOCK, 1);
        timer.Measure(version, reduce4, Grid, Block, d_a, d_out);
        cudaMemcpy(out,d_out,BLOCK_NUM*sizeof(float),cudaMemcpyDeviceToHost);

        if(check(out,res,BLOCK_NUM))printf("the ans is right\n");
        else{
            printf("the ans is wrong\n");
            for(int i=0;i<BLOCK_NUM;i++){
                printf("%lf ",out[i]);
            }
            printf("\n");
        }
    }
    else if (version == 5) {
        int NUM_PER_BLOCK = 2*THREAD_PER_BLOCK; // each thread load and add 2 data elements 
        int BLOCK_NUM = N / NUM_PER_BLOCK;
        for(int i = 0; i < BLOCK_NUM; i++){
            float cur = 0;
            for(int j = 0;j < NUM_PER_BLOCK; j++){
                cur += a[i*NUM_PER_BLOCK+j];
            }
            res[i] = cur;
        }
        dim3 Grid( BLOCK_NUM, 1);
        dim3 Block( THREAD_PER_BLOCK, 1);
        timer.Measure(version, reduce5<THREAD_PER_BLOCK>, Grid, Block, d_a, d_out);
        cudaMemcpy(out,d_out,BLOCK_NUM*sizeof(float),cudaMemcpyDeviceToHost);

        if(check(out,res,BLOCK_NUM))printf("the ans is right\n");
        else{
            printf("the ans is wrong\n");
            for(int i=0;i<BLOCK_NUM;i++){
                printf("%lf ",out[i]);
            }
            printf("\n");
        }
    }

    cudaFree(d_a);
    cudaFree(d_out);
}