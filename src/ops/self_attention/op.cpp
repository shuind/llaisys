#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_ARGUMENT(attn_val->ndim() == 3 && q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3,
                   "SelfAttention: all tensors must be 3D.");
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "SelfAttention: all tensors must be contiguous.");

    size_t L = q->shape()[0];
    size_t H = q->shape()[1];
    size_t D = q->shape()[2];
    size_t S = k->shape()[0];
    size_t HKV = k->shape()[1];
    CHECK_ARGUMENT(k->shape()[2] == D && v->shape()[2] == D, "SelfAttention: head dim mismatch.");
    CHECK_ARGUMENT(v->shape()[0] == S && v->shape()[1] == HKV, "SelfAttention: k/v shape mismatch.");
    CHECK_ARGUMENT(attn_val->shape()[0] == L && attn_val->shape()[1] == H && attn_val->shape()[2] == D,
                   "SelfAttention: output shape mismatch.");
    CHECK_ARGUMENT(H % HKV == 0, "SelfAttention: nhead must be divisible by nkvhead.");
    CHECK_ARGUMENT(S >= L, "SelfAttention: total_len must be >= seqlen.");

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(),
                                   q->data(),
                                   k->data(),
                                   v->data(),
                                   attn_val->dtype(),
                                   L, S, H, HKV, D, scale);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(),
                                   q->data(),
                                   k->data(),
                                   v->data(),
                                   attn_val->dtype(),
                                   L, S, H, HKV, D, scale);
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
