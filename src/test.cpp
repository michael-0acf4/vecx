#include "test.hpp"
#include "backend.hpp"
#include <cmath>
#include <memory>

#ifdef ENABLE_CUDA_MODE
static const char *device_name = "GPU (CUDA)";
#else
static const char *device_name = "CPU";
#endif

TEST(tests) {
  ASSERT(1 + 1 == 2);
  LGTM
}

TEST(eucl_norm_basic) {
  float data[2] = {4.0, 3.0};
  vecx v = {{2, FLOAT_32, {}}, data};

  ASSERT_CLOSE(vecx_norm(&v), 5.0, 10e-6)
  LGTM
}

TEST(eucl_norm_huge) {
  const size_t size = 65500;
  float data[size];
  for (int i = 0; i < size; data[i++] = 1)
    ;
  vecx v = {{size, FLOAT_32, {}}, data};
  ASSERT_CLOSE(vecx_norm(&v), sqrt(static_cast<double>(v.header.size)),
               _EPSILON)

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
  ASSERT_CLOSE(vecx_norm(&vx), vecx_norm(&qvx), 0.5)

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
  ASSERT_EQ(out.item_as<float>(1), 1.0)
  ASSERT_EQ(out.item_as<float>(4), 16.0)
  ASSERT_EQ(out.item_as<float>(625), 390625)

  LGTM
}

TEST(binary_operation_qi8) {
  const uint64_t size = 727;
  vecx_header qheader = {size, QINT_8, {2.847058823529412, -128}};
  std::unique_ptr<int8_t[]> qdata(new int8_t[size]);
  for (int i = 0; i < size; ++i) {
    qdata[i] = _cpu_quantize_i8(i, qheader.qparams);
  }
  vecx qvx = {qheader, qdata.get()};
  ASSERT_EQ(qvx.item_as<int8_t>(0), -128);
  ASSERT_EQ(qvx.item_as<int8_t>(size - 1), 127);
  ASSERT_CLOSE(_cpu_dequantize_i8(qvx.item_as<int8_t>(123), qheader.qparams),
               122.423531, _EPSILON);
  ASSERT_EQ(_cpu_dequantize_i8(qvx.item_as<int8_t>(size - 1), qheader.qparams),
            726.0);

  vecx_header dest_header = {size, FLOAT_32, {}};
  size_t dest_blob_size = dest_header.bytes_count_total();
  std::unique_ptr<int8_t[]> dest(new int8_t[dest_blob_size]);
  vecx_status mstatus = vecx_mult(&qvx, &qvx, dest.get());
  ASSERT_EQ(mstatus, VECX_OK)

  vecx out;
  vecx_parse_blob(dest.get(), dest_blob_size, &out);
  ASSERT_CLOSE(out.item_as<float>(4),
               (_cpu_dequantize_i8(qvx.item_as<int8_t>(4), qvx.header.qparams) *
                _cpu_dequantize_i8(qvx.item_as<int8_t>(4), qvx.header.qparams)),
               _EPSILON);

  ASSERT_EQ(out.item_as<float>(625),
            (_cpu_dequantize_i8(qvx.item_as<int8_t>(625), qvx.header.qparams) *
             _cpu_dequantize_i8(qvx.item_as<int8_t>(625), qvx.header.qparams)))

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

  vecx_header qheader = {size, QINT_8, {1.0, -128}};
  vecx qvx = {qheader, data.get()};

  int ms_norm = 0;
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    ASSERT_CLOSE(vecx_norm(&qvx), 23170.474609, _EPSILON)
    auto t2 = std::chrono::high_resolution_clock::now();
    ms_norm =
        std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  }

  int ms_dequant = 0;
  {
    // WARN: 4 * actual_size_mb more new allocations
    vecx_header header = {size, FLOAT_32, {0.0, 0}};
    size_t dest_blob_size = header.bytes_count_total();
    std::unique_ptr<uint8_t[]> blob(new uint8_t[dest_blob_size]);

    auto t1 = std::chrono::high_resolution_clock::now();
    vecx_dequantize_to_f32(&qvx, blob.get());
    auto t2 = std::chrono::high_resolution_clock::now();

    vecx vx;
    vecx_parse_blob(blob.get(), dest_blob_size, &vx);
    ASSERT_CLOSE(vecx_norm(&vx), 23170.474609, _EPSILON)

    ms_dequant =
        std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  }

  DEBUG_NUMBER(norm, ms_norm)
  DEBUG_NUMBER(dequant, ms_dequant)

  const char *runs_ci = std::getenv("GITHUB_CI");
  int delta = runs_ci ? 100 : 0;

#ifdef ENABLE_CUDA_MODE
  // RTX 3070
  ASSERT(ms_norm < 150 + delta)
  ASSERT(ms_dequant < 700 + delta)
#else
  // Core i5 11400H 2.70GHz (6 Cores)
  // SIMD is often faster for unit ops as there
  // are less memory copy overhead
  ASSERT(ms_norm < 160 + delta)
  ASSERT(ms_dequant < 600 + delta)
#endif

  LGTM
}

int main() {
  std::cout << "Device: " << yellow(device_name) << "\n";
  init_device();
  return run_all_tests();
}
