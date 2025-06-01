#include "backend.hpp"

#include <cmath>
#include <cuda_runtime.h>
#include <iostream>

// WARNING: only use when expr owns all its ressources (destroyed when out of scope)
#define UNSAFE_MAYBE_ABORT(expr)  \
    do                            \
    {                             \
        vecx_result res = (expr); \
        if (res.is_err())         \
            return res;           \
    } while (0)

#define UNSAFE_MAYBE_ABORT_CUDA(err)                                   \
    do                                                                 \
    {                                                                  \
        if (err != cudaSuccess)                                        \
            return vecx_result::device_error(cudaGetErrorString(err)); \
    } while (0)

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
            return vecx_result::device_error("device_memset: size is too large");

        if (device_data == nullptr)
            return vecx_result::device_error("device_memset: device data not initialized yet");

        cudaError_t err = cudaMemset((void *)device_data, value, device_size);
        if (err != cudaSuccess)
            return vecx_result::device_error("device_memset: " + std::string(cudaGetErrorString(err)));

        return vecx_result::ok();
    }

    vecx_result device_memset(T value) const
    {
        return device_memset(value, sizeof(T));
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

    vecx_result device2host(T *host_receiver, size_t host_receiver_size) const
    {
        if (host_receiver_size < device_size)
            return vecx_result::device_error("device2host: host buffer size is too small");

        cudaError_t err = cudaMemcpy((void *)host_receiver, (void *)device_data, device_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
            return vecx_result::device_error("device2host: " + std::string(cudaGetErrorString(err)));

        return vecx_result::ok();
    }

    T *get()
    {
        return device_data;
    }

    T *&ref()
    {
        return device_data;
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

// FIXME: use atomicAdd(double*, double) (will not work on old GPUs)
template <typename T>
__global__ void dot_kernel(T *a, T *b, uint64_t size, quant_params a_qparams, quant_params b_qparams, float *result)
{
    extern __shared__ float partial_sum[];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x;

    float sum = 0.0f;
    if (tid < size)
    {
        float va = maybe_dequantize(a[tid], a_qparams);
        float vb = maybe_dequantize(b[tid], b_qparams);
        sum += va * vb;
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
vecx_result dot_host(const vecx *a, const vecx *b, float *sum)
{
    vecx_result check = validate_layout_similarities(a, b);
    if (check != vecx_result::ok())
        return check;

    cuda_ptr<T> d_a_data;
    UNSAFE_MAYBE_ABORT(d_a_data.hostvecx2device(a));

    cuda_ptr<T> &d_b_data = d_a_data;
    if (a != b)
    {
        d_b_data = cuda_ptr<T>();
        UNSAFE_MAYBE_ABORT(d_b_data.hostvecx2device(b));
    }

    cuda_ptr<float> d_result;
    UNSAFE_MAYBE_ABORT(d_result.device_alloc(sizeof(float)));
    UNSAFE_MAYBE_ABORT(d_result.device_memset(0));

    size_t type_size = a->header.type_size();
    int threads = 256;
    int blocks = (a->header.size + threads - 1) / threads;
    size_t shared_size = threads * type_size;
    dot_kernel<<<blocks, threads, shared_size>>>(d_a_data.ref(), d_b_data.ref(), a->header.size, a->header.qparams, b->header.qparams, d_result.ref());
    UNSAFE_MAYBE_ABORT_CUDA(cudaGetLastError());
    UNSAFE_MAYBE_ABORT_CUDA(cudaDeviceSynchronize());

    UNSAFE_MAYBE_ABORT(d_result.device2host(sum, sizeof(float)));

    return vecx_result::ok();
}

vecx_result vecx_dot(const vecx *a, const vecx *b, double *sum)
{
    vecx_result check = validate_layout_similarities(a, b);
    if (check != vecx_result::ok())
        return check;

    float tsum = 0.0;

    if (a->header.dtype == FLOAT_32)
        dot_host<float>(a, b, &tsum);
    else
        dot_host<int8_t>(a, b, &tsum);

    *sum = (double)tsum;

    return vecx_result::ok();
}

double vecx_norm(const vecx *v)
{
    double sum;
    vecx_dot(v, v, &sum);
    return sqrtf(sum);
}

template <typename T, typename op_trivial>
__global__ void op_apply_kernel(T *a, T *b,
                                float *result,
                                size_t size,
                                quant_params a_qparams,
                                quant_params b_qparams,
                                op_trivial op_apply)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        float va = maybe_dequantize(a[i], a_qparams);
        float vb = maybe_dequantize(b[i], b_qparams);
        result[i] = op_apply(va, vb);
    }
}

template <typename T, typename op_trivial>
vecx_result op_apply_host(const vecx *a, const vecx *b, void *dest, op_trivial op_apply)
{
    vecx_result check = validate_layout_similarities(a, b);
    if (check != vecx_result::ok())
        return check;

    cuda_ptr<T> d_a_data;
    UNSAFE_MAYBE_ABORT(d_a_data.hostvecx2device(a));

    cuda_ptr<T> &d_b_data = d_a_data;
    if (a != b)
    {
        d_b_data = cuda_ptr<T>();
        UNSAFE_MAYBE_ABORT(d_b_data.hostvecx2device(b));
    }

    size_t size = a->header.size;
    cuda_ptr<float> d_result;
    d_result.device_alloc(size * sizeof(float));
    d_result.device_memset(0.0, size * sizeof(float));

    int threads = 256;
    int blocks = (a->header.size + threads - 1) / threads;
    size_t type_size = vecx_type_size(a->header.dtype);
    op_apply_kernel<<<blocks, threads>>>(d_a_data.ref(), d_b_data.ref(), d_result.ref(), size, a->header.qparams, b->header.qparams, op_apply);
    UNSAFE_MAYBE_ABORT_CUDA(cudaGetLastError());
    UNSAFE_MAYBE_ABORT_CUDA(cudaDeviceSynchronize());

    vecx_header header = {size, FLOAT_32, {}};
    float *h_result = (float *)vecx_pack_header_into(header, dest);
    UNSAFE_MAYBE_ABORT(d_result.device2host(h_result, size * sizeof(float)));

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
    cuda_ptr<T> d_v_data;
    UNSAFE_MAYBE_ABORT(d_v_data.hostvecx2device(v));

    size_t size = v->header.size;
    quant_params qparams = v->header.qparams;
    cuda_ptr<float> d_result;
    d_result.device_alloc(size * sizeof(float));
    d_result.device_memset(0.0, size * sizeof(float));

    int threads = 256;
    int blocks = (v->header.size + threads - 1) / threads;
    size_t type_size = vecx_type_size(v->header.dtype);
    op_apply_broadcast_scalar_kernel<<<blocks, threads>>>(d_v_data.ref(), scalar, d_result.ref(), size, qparams, op_apply);
    UNSAFE_MAYBE_ABORT_CUDA(cudaGetLastError());
    UNSAFE_MAYBE_ABORT_CUDA(cudaDeviceSynchronize());

    vecx_header header = {size, FLOAT_32, {}};
    float *h_result = (float *)vecx_pack_header_into(header, dest);
    UNSAFE_MAYBE_ABORT(d_result.device2host(h_result, size * sizeof(float)));

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

    cuda_ptr<int8_t> d_v_data;
    UNSAFE_MAYBE_ABORT(d_v_data.hostvecx2device(v));

    size_t size = v->header.size;
    quant_params qparams = v->header.qparams;
    cuda_ptr<float> d_result;
    d_result.device_alloc(size * sizeof(float));
    d_result.device_memset(0.0, size * sizeof(float));

    int threads = 256;
    int blocks = (v->header.size + threads - 1) / threads;
    size_t type_size = vecx_type_size(v->header.dtype);
    dequantize_i8_kernel<<<blocks, threads>>>(d_v_data.ref(), d_result.ref(), size, qparams);
    UNSAFE_MAYBE_ABORT_CUDA(cudaDeviceSynchronize());

    vecx_header header = {size, FLOAT_32, {}};
    float *h_result = (float *)vecx_pack_header_into(header, dest);
    UNSAFE_MAYBE_ABORT(d_result.device2host(h_result, size * sizeof(float)));

    return vecx_result::ok();
}

// CUDA context init often skew test duration without this trick
__global__ void init_kernel() {}
void init_device()
{
    init_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}
