#pragma once
#include "common.hpp"
#include <cuda_runtime.h>
#include <cmath>

// Note: assumes vecx is a plain old struct
extern "C" float f32_norm(const vecx *v);
