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

# With cuda
make test USE_CUDA=1
make build USE_CUDA=1
```

# Datatypes

- FLOAT_32 (f32)
- QINT_8 (qi8)

# Roadmap

- [x] Euclidean distance
- [ ] Basic binary ops
  - [ ] Add
  - [ ] Substract
  - [ ] Multiply
  - [ ] Division
  - [ ] Scalar multiplication (left, right)
- [ ] Inline vector definition: `vecx_inline('-4, 9, 0.00, 4.6', 'f32')`
- [ ] Explicit logical type promotion (e.g. i64 -> f32)
- [ ] Dot product
- [ ] Vector folding `vecx_fold('+' | '-' | '/' | '*', blob, init)`
