#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias);
void linear_transposed(tensor_t out, tensor_t in, tensor_t weight_t, tensor_t bias);
}
