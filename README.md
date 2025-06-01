# vecx

A simple SQLite extension that enables direct GPU and/or SIMD accelerated vector
operations.

# Building and Testing

```bash
# With SIMD (AVX2 only)
make test
make build

# With CUDA
make test USE_CUDA=1
make build USE_CUDA=1
```

# Datatypes

- FLOAT_32 (f32)
- QINT_8 (qi8)

# Roadmap

- [x] Euclidean distance `x_norm(a)`
- [x] Basic binary ops
  - [x] Add `x_add(a, b)`
  - [x] Substract `x_sub(a, b)`
  - [x] Multiply `x_mul(a, b)`
  - [x] Division `x_div(a, b)`
  - [x] Scalar multiplication `x_mulk(a, b)`
  - [x] Dot product `x_dot(a, b)`
  - [x] Cosine similarity `x_cosim(a, b)`
- [x] Dequantize from qi8 to f32 `x_dequantize(a)`
- [x] Inline vector definition:
  - `x_vec('1.0, 2.0, 3.0, 4')`
  - `x_vec('-4, 9, 0.00, 4.6', left_pad, right_pad, fill_val)`

# Binary specification

A simple plain binary that contains the size, quantization parameters and the
vector values. You can refer to [e2e/vecx_spec.py](e2e/vecx_spec.py) for a
simple numpy based serialization.

```
4 bytes -> "vecx"
4 bytes -> dtype (i32): 1 for F32, 2 for QI8
4 bytes -> i8-based quantization scale (f32), unused if dtype = 1
4 bytes -> i8-based quantization zero point (i32), unused if dtype = 1
8 bytes -> vector size
N bytes -> data region, exactly `sizeof dtype in bytes * size`
```

> [!WARNING]
>
> You may notice that the GPU approach is often slower than its SIMD
> counterpart. This is because most operations so far are all so trivial that
> they are mostly memory-bound, and copying from host to device memory takes a
> few to hundreds of milliseconds, depending on how large your vector is. It is
> still faster than any naive implementation, but SIMD often appears faster as
> it directly reads/writes from/into host memory without any copy overhead
> (which also depends on the bandwidth of the host computer).
