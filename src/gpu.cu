#include "gpu.cuh"
#include <cuda_runtime.h>
#include <math.h>

__global__ void euclidean_norm_kernel(const float *data, uint64_t size, float *result)
{
    extern __shared__ float partial_sum[];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x;

    float sum = 0.0f;
    if (tid < size)
    {
        float val = data[tid];
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

    // final result from thread 0
    if (local_tid == 0)
        atomicAdd(result, partial_sum[0]);
}

void f32_norm_cuda(const float *data, uint64_t size, float *result)
{
    float *d_data = nullptr;
    cudaMalloc(&d_data, size * sizeof(float));
    cudaMemcpy((void *)d_data, data, size * sizeof(float), cudaMemcpyHostToDevice);

    float *d_result = nullptr;
    cudaMalloc(&d_result, sizeof(float));
    cudaMemset(d_result, 0, sizeof(float));

    // Launch
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    size_t shared_size = threads * sizeof(float);
    euclidean_norm_kernel<<<blocks, threads, shared_size>>>(d_data, size, d_result);
    cudaDeviceSynchronize();

    float h_result = 0;
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree((void *)d_data);
    cudaFree(d_result);

    *result = sqrtf(h_result);
}

extern "C" float f32_norm(const vecx *v)
{
    float r;
    f32_norm_cuda((const float *)v->data, v->size, &r);
    return r;
}
