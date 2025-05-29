#pragma once
#include "common.hpp"

vecx_result vecx_dequantize_to_f32(const vecx *v, void *dest);
double vecx_norm(const vecx *v);

vecx_result vecx_add(const vecx *a, const vecx *b, void *dest);
vecx_result vecx_sub(const vecx *a, const vecx *b, void *dest);
vecx_result vecx_mult(const vecx *a, const vecx *b, void *dest);
vecx_result vecx_div(const vecx *a, const vecx *b, void *dest);

void init_device();
