#include "test.hpp"

#include "common.hpp"

#include <cmath>
#include <memory>

#ifdef ENABLE_CUDA_MODE
#include "gpu.cuh"
#else
#include "cpu.hpp"
#endif

TEST(test) {
  ASSERT(1 + 1 == 2);
  ASSERT(1 + 2 == 3);

  LGTM
}

TEST(eucl_norm_simple) {
  vecx v;
  float x[177013];
  for (int i = 0; i < 177013; i++, x[i] = 1)
    ;
  v.data = (void *)x;
  v.dtype = FLOAT_32;
  v.size = 177013;

  // DEBUG(f32_norm(&v)) // debug :: f32_norm(&v) = 420.728
  ASSERT(fabs(f32_norm(&v) - 420.7291290129553) < 10e-3)

  LGTM
}

int main() { return run_all_tests(); }
