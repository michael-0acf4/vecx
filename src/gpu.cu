#include "gpu.cuh"
#include <cuda_runtime.h>
#include <math.h>

__global__ void euclidean_norm_kernel(const vecx *v, float *result)
{
    __shared__ float partial_sum[256];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x;

    float sum = 0.0f;
    if (tid < v->size)
    {
        const float *fdata = (const float *)v->data;
        float val = fdata[tid];
        sum = val * val;
    }

    partial_sum[local_tid] = sum;
    __syncthreads();

    // fold (basic, within block)
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (local_tid < s)
            partial_sum[local_tid] += partial_sum[local_tid + s];
        __syncthreads();
    }

    // final result from thread 0 of block 0
    if (local_tid == 0 && blockIdx.x == 0)
    {
        atomicAdd(result, partial_sum[0]);
    }
}

void f32_norm_cuda(const vecx *v, float *result)
{
    vecx *d_vec = nullptr;
    float *d_result = nullptr;
    float h_result = 0;

    cudaMalloc(&d_vec, sizeof(vecx));
    cudaMemcpy(d_vec, v, sizeof(vecx), cudaMemcpyHostToDevice);

    cudaMalloc(&d_result, sizeof(float));
    cudaMemset(d_result, 0, sizeof(float));

    // Launch
    int threads = 256;
    int blocks = (v->size + threads - 1) / threads;
    euclidean_norm_kernel<<<blocks, threads>>>(d_vec, d_result);
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_vec);
    cudaFree(d_result);

    *result = sqrtf(h_result);
}

extern "C" float f32_norm(const vecx *v)
{
    float r;
    f32_norm_cuda(v, &r);
    return r;
}
