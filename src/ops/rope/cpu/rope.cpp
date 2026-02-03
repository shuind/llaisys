#include "rope.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, size_t S, size_t H, size_t D, float theta) {
    size_t half = D / 2;
    for (size_t s = 0; s < S; s++) {
        float pos = static_cast<float>(pos_ids[s]);
        for (size_t h = 0; h < H; h++) {
            size_t base = (s * H + h) * D;
            for (size_t j = 0; j < half; j++) {
                float exp = (2.0f * static_cast<float>(j)) / static_cast<float>(D);
                float angle = pos / std::pow(theta, exp);
                float c = std::cos(angle);
                float si = std::sin(angle);

                float a = llaisys::utils::cast<float>(in[base + j]);
                float b = llaisys::utils::cast<float>(in[base + j + half]);

                out[base + j] = llaisys::utils::cast<T>(a * c - b * si);
                out[base + j + half] = llaisys::utils::cast<T>(b * c + a * si);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::int64_t *pos_ids,
          llaisysDataType_t type, size_t S, size_t H, size_t D, float theta) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out),
                     reinterpret_cast<const float *>(in),
                     pos_ids,
                     S, H, D, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out),
                     reinterpret_cast<const llaisys::fp16_t *>(in),
                     pos_ids,
                     S, H, D, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out),
                     reinterpret_cast<const llaisys::bf16_t *>(in),
                     pos_ids,
                     S, H, D, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
