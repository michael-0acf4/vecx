#pragma once
#include "common.hpp"
#include <cmath>
#include <cuda_runtime.h>

// Note: assumes vecx is a plain old struct
float f32_norm(const vecx *v);
void init_device();
