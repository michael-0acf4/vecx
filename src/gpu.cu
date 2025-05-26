#include "backend.hpp"

#include <cmath>
#include <cuda_runtime.h>

template <typename T>
__device__ inline float maybe_dequantize(T value, const quant_params &)
{
    return static_cast<float>(value);
}
template <>
__device__ inline float maybe_dequantize<int8_t>(int8_t value, const quant_params &qparams)
{
    return qparams.scale *
           static_cast<float>(static_cast<int32_t>(value) - qparams.zero);
}

struct add_trivial
{
    __device__ inline float operator()(float a, float b) const { return a + b; }
};

struct sub_trivial
{
    __device__ float operator()(float a, float b) const { return a - b; }
};
struct mul_trivial
{
    __device__ float operator()(float a, float b) const { return a * b; }
};

struct div_trivial
{
    __device__ float operator()(float a, float b) const { return a / b; }
};

template <typename T>
__global__ void euclidean_norm_kernel(const T *data, uint64_t size, quant_params qparams, float *result)
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

    if (local_tid == 0)
    {
        atomicAdd(result, partial_sum[0]);
    }
}

template <typename T>
float norm_host(const vecx *v)
{
    size_t type_size = vecx_type_size(v->header.dtype);
    T *d_data = nullptr;
    cudaMalloc(&d_data, v->header.bytes_count_data_region());
    cudaMemcpy((void *)d_data, v->data, v->header.bytes_count_data_region(), cudaMemcpyHostToDevice);

    float *d_result = nullptr;
    cudaMalloc(&d_result, sizeof(float));
    cudaMemset(d_result, 0, sizeof(float));

    int threads = 256;
    int blocks = (v->header.size + threads - 1) / threads;
    size_t shared_size = threads * type_size;
    euclidean_norm_kernel<<<blocks, threads, shared_size>>>(d_data, v->header.size, v->header.qparams, d_result);
    cudaDeviceSynchronize();

    float h_result = 0;
    cudaMemcpy(&h_result, d_result, sizeof(h_result), cudaMemcpyDeviceToHost);

    cudaFree((void *)d_data);
    cudaFree(d_result);

    return sqrtf(h_result);
}

double vecx_norm(const vecx *v)
{
    return v->header.dtype == FLOAT_32 ? norm_host<float>(v)
                                       : norm_host<int8_t>(v);
}

template <typename T, typename op_trivial>
__global__ void op_apply_kernel(T *a, T *b,
                                float *result,
                                size_t size,
                                quant_params qparams,
                                op_trivial op_apply)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        float va = maybe_dequantize(a[i], qparams);
        float vb = maybe_dequantize(b[i], qparams);
        result[i] = op_apply(va, vb);
    }
}

template <typename T, typename op_trivial>
vecx_status op_apply_host(const vecx *a, const vecx *b, void *dest, op_trivial op_apply)
{
    vecx_status check = validate_layout_similarities(a, b);
    if (check != VECX_OK)
        return check;

    T *d_a_data = nullptr;
    cudaMalloc(&d_a_data, a->header.bytes_count_data_region());
    cudaMemcpy((void *)d_a_data, a->data, a->header.bytes_count_data_region(), cudaMemcpyHostToDevice);

    T *d_b_data = nullptr;
    cudaMalloc(&d_b_data, b->header.bytes_count_data_region());
    cudaMemcpy((void *)d_b_data, b->data, b->header.bytes_count_data_region(), cudaMemcpyHostToDevice);

    size_t size = a->header.size;
    quant_params qparams = a->header.qparams;
    float *d_result = nullptr;
    cudaMalloc(&d_result, size * sizeof(float));
    cudaMemset(d_result, 0.0, size * sizeof(float));

    int threads = 256;
    int blocks = (a->header.size + threads - 1) / threads;
    size_t type_size = vecx_type_size(a->header.dtype);
    op_apply_kernel<<<blocks, threads>>>(d_a_data, d_b_data, d_result, size, qparams, op_apply);
    cudaDeviceSynchronize();

    vecx_header header = {size, FLOAT_32, {}};
    float *h_result = (float *)vecx_pack_header_into(header, dest);
    cudaMemcpy(h_result, d_result, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree((void *)d_a_data);
    cudaFree((void *)d_b_data);
    cudaFree(d_result);

    return VECX_OK;
}

vecx_status vecx_add(const vecx *a, const vecx *b, void *dest)
{
    return a->header.dtype == FLOAT_32 ? op_apply_host<float>(a, b, dest, add_trivial())
                                       : op_apply_host<int8_t>(a, b, dest, add_trivial());
}

vecx_status vecx_sub(const vecx *a, const vecx *b, void *dest)
{
    return a->header.dtype == FLOAT_32 ? op_apply_host<float>(a, b, dest, sub_trivial())
                                       : op_apply_host<int8_t>(a, b, dest, sub_trivial());
}

vecx_status vecx_mult(const vecx *a, const vecx *b, void *dest)
{
    return a->header.dtype == FLOAT_32 ? op_apply_host<float>(a, b, dest, mul_trivial())
                                       : op_apply_host<int8_t>(a, b, dest, mul_trivial());
}

vecx_status vecx_div(const vecx *a, const vecx *b, void *dest)
{
    return a->header.dtype == FLOAT_32 ? op_apply_host<float>(a, b, dest, div_trivial())
                                       : op_apply_host<int8_t>(a, b, dest, div_trivial());
}

__global__ void dequantize_i8_kernel(int8_t *a,
                                     float *result,
                                     size_t size,
                                     quant_params qparams)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        result[i] = maybe_dequantize(a[i], qparams);
    }
}

vecx_status vecx_dequantize_to_f32(const vecx *v, void *dest)
{
    if (v->header.dtype == FLOAT_32)
    {
        vecx_pack_into(*v, dest);
        return VECX_OK;
    }

    int8_t *d_v_data = nullptr;
    cudaMalloc(&d_v_data, v->header.bytes_count_data_region());
    cudaMemcpy((void *)d_v_data, v->data, v->header.bytes_count_data_region(), cudaMemcpyHostToDevice);

    size_t size = v->header.size;
    quant_params qparams = v->header.qparams;
    float *d_result = nullptr;
    cudaMalloc(&d_result, size * sizeof(float));
    cudaMemset(d_result, 0.0, size * sizeof(float));

    int threads = 256;
    int blocks = (v->header.size + threads - 1) / threads;
    size_t type_size = vecx_type_size(v->header.dtype);
    dequantize_i8_kernel<<<blocks, threads>>>(d_v_data, d_result, size, qparams);
    cudaDeviceSynchronize();

    vecx_header header = {size, FLOAT_32, {}};
    float *h_result = (float *)vecx_pack_header_into(header, dest);
    cudaMemcpy(h_result, d_result, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree((void *)d_v_data);
    cudaFree(d_result);

    return VECX_OK;
}

// CUDA context init often skew test duration without this trick
__global__ void init_kernel() {}
void init_device()
{
    init_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}
