#include "embedding.hpp"
#include "../../../utils.hpp"
#include <cstring>

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, size_t N, size_t D){
    for(size_t i = 0; i < N; i++ ) {
        int64_t row = index[i];
        const T *src =  weight + (size_t)row * D;
        T *dst = out + i * D;
        std::memcpy(dst, src, D * sizeof(T));
    }

}

namespace llaisys::ops::cpu {
void embedding(std::byte* out, const int64_t* index, const std::byte* weight, llaisysDataType_t type, size_t N, size_t D) {
    switch (type) {
        case LLAISYS_DTYPE_F32:
            return embedding_(reinterpret_cast<float*>(out),
                            index,
                            reinterpret_cast<const float*>(weight),
                            N, D);
        case LLAISYS_DTYPE_F16:
            return embedding_(reinterpret_cast<llaisys::fp16_t*>(out),
                            index,
                            reinterpret_cast<const llaisys::fp16_t*>(weight),
                            N, D);
        case LLAISYS_DTYPE_BF16:
            return embedding_(reinterpret_cast<llaisys::bf16_t*>(out),
                            index,
                            reinterpret_cast<const llaisys::bf16_t*>(weight),
                            N, D);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }
    }
}
