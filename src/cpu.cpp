#include "cpu.hpp"
#include <cmath>

double f32_norm(const vecx *v) {
  float s = 0;

  for (uint64_t i = 0; i < v->size; i++) {
    float val = 0;

    switch (v->dtype) {
    case FLOAT_32: {
      const float *data = (const float *)v->data;
      val = (double)data[i];
      break;
    }
    case INT_32: {
      const int32_t *data = (const int32_t *)v->data;
      val = (double)data[i];
      break;
    }
    case INT_64: {
      const int64_t *data = (const int64_t *)v->data;
      val = (double)data[i];
      break;
    }
    default:
      return 0.0f;
    }

    s += val * val;
  }

  return sqrt(s);
}