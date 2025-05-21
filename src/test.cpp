#include "test.hpp"
#include "common.hpp"
#include <cmath>
#include <memory>

#ifdef ENABLE_CUDA_MODE
#include "gpu.cuh"
static const char *device_name = "GPU (CUDA)";
#else
#include "cpu.hpp"
static const char *device_name = "CPU";
#endif

TEST(tests) {
  ASSERT(1 + 1 == 2);
  LGTM
}

TEST(eucl_norm_basic) {
  float data[2] = {4.0, 3.0};
  vecx v = {2, FLOAT_32, {}, data};

  ASSERT_CLOSE(f32_norm(&v), 5.0, 10e-6)
  LGTM
}

TEST(eucl_norm_huge) {
  float data[177013];
  for (int i = 0; i < 177013; data[i++] = 1)
    ;
  vecx v = {177013, FLOAT_32, {}, data};
  // DEBUG_NUMBER(eucl_norm_huge, f32_norm(&v))
  ASSERT_CLOSE(f32_norm(&v), sqrt(static_cast<double>(v.size)), _EPSILON)

  LGTM
}

TEST(dequantize) {
  quant_params qparams = {0.03529411764705882, -128};

  ASSERT_CLOSE(_cpu_dequantize_i8(-100, qparams), 1.0, 0.1)
  ASSERT_CLOSE(_cpu_dequantize_i8(-71, qparams), 2.0, 0.1)
  ASSERT_CLOSE(_cpu_dequantize_i8(-43, qparams), 3.0, 0.1)

  LGTM
}

TEST(eucl_norm_on_quantized_i8) {
  float xs[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int8_t qxs[10] = {-100, -71, -43, -15, 14, 42, 70, 99, 127, 127};
  quant_params qparams = {0.03529411764705882, -128};

  for (int i = 0; i < 10; ++i) {
    ASSERT_CLOSE(_cpu_dequantize_i8(qxs[i], qparams), xs[i], 1.3)
  }

  vecx vx = {10, FLOAT_32, {}, xs};
  vecx qvx = {10, QINT_8, qparams, qxs};
  ASSERT_CLOSE(f32_norm(&vx), f32_norm(&qvx), 0.5)

  LGTM
}

int main() {
  std::cout << "Device: " << yellow(device_name) << "\n";
  init_device();
  return run_all_tests();
}
