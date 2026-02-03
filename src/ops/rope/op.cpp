#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_ARGUMENT(out->ndim() == 3 && in->ndim() == 3, "RoPE: out/in must be 3D.");
    CHECK_ARGUMENT(pos_ids->ndim() == 1, "RoPE: pos_ids must be 1D.");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_ARGUMENT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE: pos_ids must be int64.");
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), "RoPE: all tensors must be contiguous.");

    CHECK_SAME_SHAPE(out->shape(), in->shape());
    size_t S = in->shape()[0];
    size_t H = in->shape()[1];
    size_t D = in->shape()[2];
    CHECK_ARGUMENT(D % 2 == 0, "RoPE: head_dim must be even.");
    CHECK_ARGUMENT(pos_ids->shape()[0] == S, "RoPE: pos_ids length mismatch.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(),
                         in->data(),
                         reinterpret_cast<const int64_t *>(pos_ids->data()),
                         out->dtype(),
                         S, H, D, theta);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(),
                         in->data(),
                         reinterpret_cast<const int64_t *>(pos_ids->data()),
                         out->dtype(),
                         S, H, D, theta);
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
