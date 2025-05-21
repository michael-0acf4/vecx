#include "gpu.cuh"
#include <cuda_runtime.h>
#include <math.h>

template <typename T>
__device__ inline float maybe_dequantize(T value, const quant_params &)
{
    return static_cast<float>(value);
}
// Fallback
template <>
__device__ inline float maybe_dequantize<int8_t>(int8_t value, const quant_params &qparams)
{
    return qparams.scale *
           static_cast<float>(static_cast<int32_t>(value) - qparams.zero);
}

template <typename T>
__global__ void device_euclidean_norm_kernel(const T *data, uint64_t size, quant_params qparams, float *result)
{
    extern __shared__ float partial_sum[];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x;

    float sum = 0.0f;
    if (tid < size)
    {
        float val = maybe_dequantize(data[tid], qparams);
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

template <typename T>
float f32_norm_host(const vecx *v)
{
    size_t type_size = vecx_type_size(v->dtype);
    T *d_data = nullptr;
    cudaMalloc(&d_data, v->size * type_size);
    cudaMemcpy((void *)d_data, v->data, v->size * type_size, cudaMemcpyHostToDevice);

    float *d_result = nullptr;
    cudaMalloc(&d_result, sizeof(float));
    cudaMemset(d_result, 0, sizeof(float));

    // Launch
    int threads = 256;
    int blocks = (v->size + threads - 1) / threads;
    size_t shared_size = threads * type_size;
    device_euclidean_norm_kernel<<<blocks, threads, shared_size>>>(d_data, v->size, v->qparams, d_result);
    cudaDeviceSynchronize();

    float h_result = 0;
    cudaMemcpy(&h_result, d_result, sizeof(h_result), cudaMemcpyDeviceToHost);

    cudaFree((void *)d_data);
    cudaFree(d_result);

    return sqrtf(h_result);
}

float f32_norm(const vecx *v)
{
    return v->dtype == FLOAT_32 ? f32_norm_host<float>(v) : f32_norm_host<int8_t>(v);
}

// CUDA context init often skew test duration without this trick
__global__ void init_kernel() {}
void init_device()
{
    init_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}
