#include "backend.hpp"

#include <cmath>
#include <cuda_runtime.h>
#include <iostream>

// WARNING: only use when expr owns all its ressources (destroyed when out of scope)
#define UNSAFE_MAYBE_ABORT(expr)                         \
    {                                                    \
        vecx_result res = (expr);                        \
        if (res.is_err())                                \
        {                                                \
            /* std::cout << res.error_payload << "\n";*/ \
            return res;                                  \
        }                                                \
    }

// RAII style custom unique pointer
// This enables safe early returns with device allocated memory

template <typename T>
class cuda_ptr
{
public:
    T *device_data = nullptr;
    size_t device_size = 0;

    ~cuda_ptr()
    {
        cudaError_t err = cudaFree((void *)device_data);
        // std::cout << "Freed " << cudaGetErrorString(err) << ", size" << device_size << "\n";
        device_data = nullptr;
        device_size = 0;
    }

    vecx_result device_memset(T value, size_t size) const
    {
        if (device_size < size)
            return vecx_result::device_error("device_memset: size is too large during");

        if (device_data == nullptr)
            return vecx_result::device_error("device_memset: device data not initialized yet during");

        cudaError_t err = cudaMemset((void *)device_data, value, device_size);
        if (err != cudaSuccess)
            return vecx_result::device_error("device_memset: " + std::string(cudaGetErrorString(err)));

        return vecx_result::ok();
    }

    vecx_result device_memset(T value) const
    {
        return device_memset(value, 1);
    }

    vecx_result device_alloc(size_t size)
    {
        if (device_size != 0 || device_data != nullptr)
            return vecx_result::device_error("device buffer already allocated");

        if (size <= 0)
            return vecx_result::device_error("host size cannot be less than or equal to 0");

        device_size = size;
        cudaError_t err = cudaSuccess;

        err = cudaMalloc(&device_data, size);
        if (err != cudaSuccess)
            return vecx_result::device_error("device_alloc: " + std::string(cudaGetErrorString(err)));

        return vecx_result::ok();
    }

    vecx_result hostvecx2device(const vecx *v)
    {
        T *data = (T *)v->data_as<T>();
        return host2device(data, v->header.bytes_count_data_region());
    }

    vecx_result host2device(T *host_data, size_t host_size)
    {
        vecx_result res = device_alloc(host_size);
        if (res.is_err())
            return res;

        cudaError_t err = cudaMemcpy((void *)device_data, (void *)host_data, host_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
            return vecx_result::device_error("host2device: " + std::string(cudaGetErrorString(err)));

        return vecx_result::ok();
    }

    vecx_result device2host(T *host_receiver, size_t host_rc_size) const
    {
        cudaError_t err = cudaMemcpy((void *)host_receiver, (void *)device_data, device_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
            return vecx_result::device_error("device2host: " + std::string(cudaGetErrorString(err)));

        return vecx_result::ok();
    }
};

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

    cuda_ptr<T> d_data;
    UNSAFE_MAYBE_ABORT(d_data.hostvecx2device(v));

    cuda_ptr<float> d_result;
    UNSAFE_MAYBE_ABORT(d_result.device_alloc(sizeof(float)));
    UNSAFE_MAYBE_ABORT(d_result.device_memset(0));

    size_t type_size = v->header.type_size();
    int threads = 256;
    int blocks = (v->header.size + threads - 1) / threads;
    size_t shared_size = threads * type_size;
    euclidean_norm_kernel<<<blocks, threads, shared_size>>>(d_data.device_data, v->header.size, v->header.qparams, d_result.device_data);
    cudaDeviceSynchronize();

    float h_result = 0;
    UNSAFE_MAYBE_ABORT(d_result.device2host(&h_result, sizeof(float)));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        return vecx_result::device_error(cudaGetErrorString(err));
    }

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
vecx_result op_apply_host(const vecx *a, const vecx *b, void *dest, op_trivial op_apply)
{
    vecx_result check = validate_layout_similarities(a, b);
    if (check != vecx_result::ok())
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

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        return vecx_result::device_error(cudaGetErrorString(err));
    }

    return vecx_result::ok();
}

template <typename T, typename op_trivial>
__global__ void op_apply_broadcast_scalar_kernel(T *v, float scalar,
                                                 float *result,
                                                 size_t size,
                                                 quant_params qparams,
                                                 op_trivial op_apply)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        float va = maybe_dequantize(v[i], qparams);
        result[i] = op_apply(va, scalar);
    }
}

template <typename T, typename op_trivial>
vecx_result op_apply_broadcast_scalar_host(const vecx *v, float scalar, void *dest, op_trivial op_apply)
{
    T *d_v_data = nullptr;
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
    op_apply_broadcast_scalar_kernel<<<blocks, threads>>>(d_v_data, scalar, d_result, size, qparams, op_apply);
    cudaDeviceSynchronize();

    vecx_header header = {size, FLOAT_32, {}};
    float *h_result = (float *)vecx_pack_header_into(header, dest);
    cudaMemcpy(h_result, d_result, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree((void *)d_v_data);
    cudaFree(d_result);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        return vecx_result::device_error(cudaGetErrorString(err));
    }

    return vecx_result::ok();
}

vecx_result vecx_add(const vecx *a, const vecx *b, void *dest)
{
    return a->header.dtype == FLOAT_32 ? op_apply_host<float>(a, b, dest, add_trivial())
                                       : op_apply_host<int8_t>(a, b, dest, add_trivial());
}

vecx_result vecx_sub(const vecx *a, const vecx *b, void *dest)
{
    return a->header.dtype == FLOAT_32 ? op_apply_host<float>(a, b, dest, sub_trivial())
                                       : op_apply_host<int8_t>(a, b, dest, sub_trivial());
}

vecx_result vecx_mult(const vecx *a, const vecx *b, void *dest)
{
    return a->header.dtype == FLOAT_32 ? op_apply_host<float>(a, b, dest, mul_trivial())
                                       : op_apply_host<int8_t>(a, b, dest, mul_trivial());
}

vecx_result vecx_div(const vecx *a, const vecx *b, void *dest)
{
    return a->header.dtype == FLOAT_32 ? op_apply_host<float>(a, b, dest, div_trivial())
                                       : op_apply_host<int8_t>(a, b, dest, div_trivial());
}

vecx_result vecx_scalar(const vecx *v, float scalar, void *dest)
{
    return v->header.dtype == FLOAT_32 ? op_apply_broadcast_scalar_host<float>(v, scalar, dest, mul_trivial())
                                       : op_apply_broadcast_scalar_host<int8_t>(v, scalar, dest, mul_trivial());
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

vecx_result vecx_dequantize_to_f32(const vecx *v, void *dest)
{
    if (v->header.dtype == FLOAT_32)
    {
        vecx_pack_into(*v, dest);
        return vecx_result::ok();
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

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        return vecx_result::device_error(cudaGetErrorString(err));
    }

    return vecx_result::ok();
}

// CUDA context init often skew test duration without this trick
__global__ void init_kernel() {}
void init_device()
{
    init_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}
