#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_ARGUMENT(index->ndim() == 1, "Embedding: index must be 1D.");
    CHECK_ARGUMENT(weight->ndim() == 2, "Embedding: weight must be 2D.");
    CHECK_ARGUMENT(out->ndim() == 2, "Embedding: out must be 2D.");
    CHECK_ARGUMENT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index must be int64.");
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), "Embedding: all tensors must be contiguous.");

    size_t N = index->numel();
    size_t D = weight->shape()[1];
    CHECK_ARGUMENT(out->shape()[0] == N && out->shape()[1] == D, "Embedding: out shape mismatch.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(),
                              reinterpret_cast<const int64_t *>(index->data()),
                              weight->data(),
                              out->dtype(),
                              N,
                              D);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(),
                              reinterpret_cast<const int64_t *>(index->data()),
                              weight->data(),
                              out->dtype(),
                              N,
                              D);
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
