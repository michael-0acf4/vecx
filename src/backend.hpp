#pragma once
#include "common.hpp"

vecx_status vecx_dequantize_to_f32(const vecx *v, void *dest);
double f32_norm(const vecx *v);

vecx_status vecx_add(const vecx *a, const vecx *b, void *dest);
vecx_status vecx_sub(const vecx *a, const vecx *b, void *dest);
vecx_status vecx_mult(const vecx *a, const vecx *b, void *dest);
vecx_status vecx_div(const vecx *a, const vecx *b, void *dest);

void init_device();
