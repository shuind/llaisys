#include "self_attention.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <limits>
#include <vector>

template <typename T>
void self_attention_(T *attn_val,
                     const T *q,
                     const T *k,
                     const T *v,
                     size_t L,
                     size_t S,
                     size_t H,
                     size_t HKV,
                     size_t D,
                     float scale) {
    const size_t head_repeat = H / HKV;
    const size_t mask_offset = S - L;

    std::vector<float> out_acc(D);

    for (size_t h = 0; h < H; h++) {
        const size_t kvh = h / head_repeat;

        for (size_t i = 0; i < L; i++) {
            const size_t max_j = std::min(i + mask_offset, S - 1);

            const T *q_vec = q + (i * H + h) * D;
            T *out_vec = attn_val + (i * H + h) * D;

            float m = -std::numeric_limits<float>::infinity(); // running max
            float l = 0.0f;                                    // running denom (sum exp)
            std::fill(out_acc.begin(), out_acc.end(), 0.0f);   // running numerator vector

            for (size_t j = 0; j <= max_j; j++) {
                const T *k_vec = k + (j * HKV + kvh) * D;

                float dot = 0.0f;
                for (size_t d = 0; d < D; d++) {
                    dot += llaisys::utils::cast<float>(q_vec[d]) *
                           llaisys::utils::cast<float>(k_vec[d]);
                }
                const float s = dot * scale;

                const T *v_vec = v + (j * HKV + kvh) * D;

                if (s > m) {
                    const float factor = std::exp(m - s); // in (0,1]
                    // out_acc *= factor
                    for (size_t d = 0; d < D; d++) {
                        out_acc[d] *= factor;
                    }
                    l *= factor;

                    m = s;
                    l += 1.0f;
                    for (size_t d = 0; d < D; d++) {
                        out_acc[d] += llaisys::utils::cast<float>(v_vec[d]);
                    }
                } else {
                    const float w = std::exp(s - m);
                    l += w;
                    for (size_t d = 0; d < D; d++) {
                        out_acc[d] += w * llaisys::utils::cast<float>(v_vec[d]);
                    }
                }
            }

            const float inv_l = 1.0f / l;
            for (size_t d = 0; d < D; d++) {
                out_vec[d] = llaisys::utils::cast<T>(out_acc[d] * inv_l);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val,
                    const std::byte *q,
                    const std::byte *k,
                    const std::byte *v,
                    llaisysDataType_t type,
                    size_t L,
                    size_t S,
                    size_t H,
                    size_t HKV,
                    size_t D,
                    float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val),
                               reinterpret_cast<const float *>(q),
                               reinterpret_cast<const float *>(k),
                               reinterpret_cast<const float *>(v),
                               L, S, H, HKV, D, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val),
                               reinterpret_cast<const llaisys::fp16_t *>(q),
                               reinterpret_cast<const llaisys::fp16_t *>(k),
                               reinterpret_cast<const llaisys::fp16_t *>(v),
                               L, S, H, HKV, D, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val),
                               reinterpret_cast<const llaisys::bf16_t *>(q),
                               reinterpret_cast<const llaisys::bf16_t *>(k),
                               reinterpret_cast<const llaisys::bf16_t *>(v),
                               L, S, H, HKV, D, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
