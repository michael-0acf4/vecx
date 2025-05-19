#pragma once
#include "common.hpp"
#include <cmath>
#include <cuda_runtime.h>

// Note: assumes vecx is a plain old struct
extern "C" float f32_norm(const vecx *v);
