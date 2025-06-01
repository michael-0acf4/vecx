# vecx

A simple SQLite extension that enables direct GPU and/or SIMD accelerated vector
operations.

> [!WARNING]
>
> This is still at the draft stage.

# Building and Testing

```bash
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
- [ ] Inline vector definition:
  - `x_vec('1.0, 2.0, 3.0, 4')`
  - `x_vecpad('-4, 9, 0.00, 4.6', fill_left, fill_right)`
