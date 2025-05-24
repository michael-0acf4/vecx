#pragma once
#include "common.hpp"

double f32_norm(const vecx *v);
double f32_dot_float_x_quant(const vecx *a_float, const vecx *b_quant);

void init_device();
