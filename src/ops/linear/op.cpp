#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias) {
        CHECK_SAME_DEVICE(out, bias);
    }
    CHECK_ARGUMENT(out->ndim() == 2 && in->ndim() == 2 && weight->ndim() == 2, "Linear: out/in/weight must be 2D.");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    if (bias) {
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
    }
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "Linear: out/in/weight must be contiguous.");
    if (bias) {
        ASSERT(bias->isContiguous(), "Linear: bias must be contiguous.");
    }

    size_t M = in->shape()[0];
    size_t K = in->shape()[1];
    size_t N = weight->shape()[0];
    CHECK_ARGUMENT(weight->shape()[1] == K, "Linear: weight shape mismatch.");
    CHECK_ARGUMENT(out->shape()[0] == M && out->shape()[1] == N, "Linear: out shape mismatch.");
    if (bias) {
        CHECK_ARGUMENT(bias->ndim() == 1 && bias->shape()[0] == N, "Linear: bias shape mismatch.");
    }

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(),
                           in->data(),
                           weight->data(),
                           bias ? bias->data() : nullptr,
                           out->dtype(),
                           M, N, K,
                           false);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(),
                           in->data(),
                           weight->data(),
                           bias ? bias->data() : nullptr,
                           out->dtype(),
                           M, N, K,
                           false);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

void linear_transposed(tensor_t out, tensor_t in, tensor_t weight_t, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight_t);
    if (bias) {
        CHECK_SAME_DEVICE(out, bias);
    }
    CHECK_ARGUMENT(out->ndim() == 2 && in->ndim() == 2 && weight_t->ndim() == 2, "Linear: out/in/weight must be 2D.");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight_t->dtype());
    if (bias) {
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
    }
    ASSERT(out->isContiguous() && in->isContiguous() && weight_t->isContiguous(),
           "Linear: out/in/weight must be contiguous.");
    if (bias) {
        ASSERT(bias->isContiguous(), "Linear: bias must be contiguous.");
    }

    size_t M = in->shape()[0];
    size_t K = in->shape()[1];
    size_t N = weight_t->shape()[1];
    CHECK_ARGUMENT(weight_t->shape()[0] == K, "Linear: weight shape mismatch.");
    CHECK_ARGUMENT(out->shape()[0] == M && out->shape()[1] == N, "Linear: out shape mismatch.");
    if (bias) {
        CHECK_ARGUMENT(bias->ndim() == 1 && bias->shape()[0] == N, "Linear: bias shape mismatch.");
    }

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(),
                           in->data(),
                           weight_t->data(),
                           bias ? bias->data() : nullptr,
                           out->dtype(),
                           M, N, K,
                           true);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(),
                           in->data(),
                           weight_t->data(),
                           bias ? bias->data() : nullptr,
                           out->dtype(),
                           M, N, K,
                           true);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
