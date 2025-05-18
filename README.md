# vecx

A simple SQLite extension that enables direct GPU accelerated vector operations.

# Building

```bash
make build

# Using cuda
make build USE_CUDA=1
```

# Roadmap

- [ ] Euclidean distance
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
- [ ] Matrix multiplication (+reshape)
  - [ ] `vecx_matmul(vecx_reshape(col1, 3, 3), vecx_reshape(col1, 3, 1))`
