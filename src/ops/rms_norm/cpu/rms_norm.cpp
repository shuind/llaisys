#include "rms_norm.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t M, size_t K, float eps) {
    for (size_t i = 0; i < M; i++) {
        const T *in_row = in + i * K;
        T *out_row = out + i * K;

        float mean_sq = 0.0f;
        for (size_t j = 0; j < K; j++) {
            float v = llaisys::utils::cast<float>(in_row[j]);
            mean_sq += v * v;
        }
        mean_sq /= static_cast<float>(K);
        float inv_rms = 1.0f / std::sqrt(mean_sq + eps);

        for (size_t j = 0; j < K; j++) {
            float v = llaisys::utils::cast<float>(in_row[j]);
            float w = llaisys::utils::cast<float>(weight[j]);
            out_row[j] = llaisys::utils::cast<T>(v * inv_rms * w);
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t type,
              size_t M, size_t K, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out),
                         reinterpret_cast<const float *>(in),
                         reinterpret_cast<const float *>(weight),
                         M, K, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out),
                         reinterpret_cast<const llaisys::fp16_t *>(in),
                         reinterpret_cast<const llaisys::fp16_t *>(weight),
                         M, K, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out),
                         reinterpret_cast<const llaisys::bf16_t *>(in),
                         reinterpret_cast<const llaisys::bf16_t *>(weight),
                         M, K, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
