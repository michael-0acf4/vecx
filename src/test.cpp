#include "test.hpp"

#include "common.hpp"

#include <cmath>
#include <memory>

#ifdef ENABLE_CUDA_MODE
#include "gpu.cuh"
#define IS_GPU 1
#else
#include "cpu.hpp"
#endif

TEST(tests) {
  ASSERT(1 + 1 == 2);
  LGTM
}

TEST(eucl_norm_basic) {
  float data[2] = {4.0, 3.0};
  vecx v = {.size = 2, .dtype = FLOAT_32, .data = data};

  ASSERT(fabs(f32_norm(&v) - 5.0) < 10e-6)

  LGTM
}

TEST(eucl_norm_huge) {
  float data[177013];
  for (int i = 0; i < 177013; data[i++] = 1)
    ;
  vecx v = {.size = 177013, .dtype = FLOAT_32, .data = data};

  // DEBUG_NUMBER(f32_norm(&v) -
  //              sqrt(v.size))
  ASSERT(fabs(f32_norm(&v) - sqrt(static_cast<double>(v.size))) < 10e-6)

  LGTM
}

int main() { return run_all_tests(); }
