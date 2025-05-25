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
  vecx v = {{2, FLOAT_32, {}}, data};

  ASSERT_CLOSE(f32_norm(&v), 5.0, 10e-6)
  LGTM
}

TEST(eucl_norm_huge) {
  const size_t size = 65500;
  float data[size];
  for (int i = 0; i < size; data[i++] = 1)
    ;
  vecx v = {{size, FLOAT_32, {}}, data};
  ASSERT_CLOSE(f32_norm(&v), sqrt(static_cast<double>(v.header.size)), _EPSILON)

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

  vecx vx = {{10, FLOAT_32, {}}, xs};
  vecx qvx = {{10, QINT_8, qparams}, qxs};
  ASSERT_CLOSE(f32_norm(&vx), f32_norm(&qvx), 0.5)

  LGTM
}

TEST(utils_speed_packing) {
  float xs[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  vecx_header header = {10, FLOAT_32, {}};
  vecx vx = {header, xs};

  size_t dest_blob_size = header.bytes_count_total();
  std::unique_ptr<uint8_t[]> blob(new uint8_t[dest_blob_size]);
  vecx_pack_into(vx, blob.get());

  vecx same_vx;
  vecx_status status =
      vecx_parse_blob(blob.get(), header.bytes_count_total(), &same_vx);
  ASSERT_EQ(status, VECX_OK)

  ASSERT(vx.header.dtype == same_vx.header.dtype);
  ASSERT(vx.header.size == same_vx.header.size);

  LGTM
}

TEST(binary_operation_f32) {
  const uint64_t size = 727;
  vecx_header header = {size, FLOAT_32, {}};
  std::unique_ptr<float[]> data(new float[size]);
  for (int i = 0; i < size; ++i) {
    data[i] = 1.0 * i;
  }
  vecx vx = {header, data.get()};

  size_t dest_blob_size = header.bytes_count_total();
  std::unique_ptr<int8_t[]> dest(new int8_t[dest_blob_size]);

  vecx_status mstatus = vecx_mult(&vx, &vx, dest.get());
  ASSERT_EQ(mstatus, VECX_OK)

  vecx out;
  vecx_parse_blob(dest.get(), dest_blob_size, &out);
  ASSERT_EQ(out.item_as<float>(4), 16.0)
  ASSERT_EQ(out.item_as<float>(625), 390625)

  LGTM
}

TEST(binary_operation_qi8) {
  const uint64_t size = 727;
  vecx_header qheader = {size, QINT_8, {}};
  std::unique_ptr<int8_t[]> qdata(new int8_t[size]);
  for (int i = 0; i < size; ++i) {
    qdata[i] = _cpu_quantize_i8(1.0 * i, {2.847058823529412, -128});
  }
  vecx qvx = {qheader, qdata.get()};

  // qi8 x qi8
  vecx_header dest_header = {size, FLOAT_32, {}};
  size_t dest_blob_size = dest_header.bytes_count_total();
  std::unique_ptr<int8_t[]> dest(new int8_t[dest_blob_size]);
  vecx_status mstatus = vecx_mult(&qvx, &qvx, dest.get());
  ASSERT_EQ(mstatus, VECX_OK)

  // vecx out;
  // vecx_parse_blob(dest.get(), dest_blob_size, &out);
  // ASSERT_EQ(out.item_as<float>(4), 16.0)
  // ASSERT_EQ(out.item_as<float>(625), 390625)

  LGTM
}

TEST(eucl_norm_on_huge_quantized_i8) {
  // Tensors often are around 100mb, 200mb, 1Gb
  const uint64_t actual_size_mb = 512u;
  const uint64_t size = actual_size_mb * 1048576u;
  // stack is too small for large vectors
  std::unique_ptr<int8_t[]> data(new int8_t[size]);

  for (int i = 0; i < size; ++i) {
    data[i] = _cpu_quantize_i8(1.0, {1.0, -128});
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  vecx_header qheader = {size, QINT_8, {1.0, -128}};
  vecx qvx = {qheader, data.get()};
  ASSERT_CLOSE(f32_norm(&qvx), 23170.474609, _EPSILON)

  // WARN: 4 * actual_size_mb more new allocations
  vecx_header header = {size, FLOAT_32, {0.0, 0}};
  size_t dest_blob_size = header.bytes_count_total();
  std::unique_ptr<uint8_t[]> blob(new uint8_t[dest_blob_size]);
  vecx_dequantize_to_f32(qvx, blob.get());

  vecx vx;
  vecx_parse_blob(blob.get(), dest_blob_size, &vx);

  ASSERT_EQ(vx.header.size, size)
  ASSERT_EQ(vx.header.dtype, FLOAT_32)
  ASSERT_EQ(vx.header.qparams.zero, 0)

  ASSERT_CLOSE(((float *)vx.data)[0], 1.0, _EPSILON)
  ASSERT_CLOSE(f32_norm(&vx), 23170.474609, _EPSILON)
  auto t2 = std::chrono::high_resolution_clock::now();

  auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
  DEBUG_NUMBER(duration, ms_int.count())

#ifdef ENABLE_CUDA_MODE
  // RTX 3070
  ASSERT(ms_int.count() < 900)
#else
  // SIMD is faster for reduce ops
  ASSERT(ms_int.count() < 800)
#endif

  LGTM
}

int main() {
  std::cout << "Device: " << yellow(device_name) << "\n";
  init_device();
  return run_all_tests();
}
