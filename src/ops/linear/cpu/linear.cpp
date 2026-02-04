#include "linear.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <thread>
#include <vector>

template <typename T>
void linear_nt(T *out, const T *in, const T *weight, const T *bias, size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; i++) {
        const T *in_row = in + i * K;
        T *out_row = out + i * N;
        for (size_t j = 0; j < N; j++) {
            const T *w_row = weight + j * K;
            float acc = (bias != nullptr) ? llaisys::utils::cast<float>(bias[j]) : 0.0f;
            const T *in_ptr = in_row;
            const T *w_ptr = w_row;
            for (size_t k = 0; k < K; k++) {
                acc += llaisys::utils::cast<float>(*in_ptr++) * llaisys::utils::cast<float>(*w_ptr++);
            }
            out_row[j] = llaisys::utils::cast<T>(acc);
        }
    }
}

template <typename T>
void linear_t(T *out, const T *in, const T *weight_t, const T *bias, size_t M, size_t N, size_t K) {
    size_t nt = std::thread::hardware_concurrency();
    if (nt == 0) {
        nt = 4;
    }
    if (nt > M) {
        nt = M;
    }
    if (nt <= 1 || (M * K) < 4096) {
        std::vector<float> acc_row(N);
        for (size_t i = 0; i < M; i++) {
            const T *in_row = in + i * K;
            T *out_row = out + i * N;
            if (bias != nullptr) {
                for (size_t j = 0; j < N; j++) {
                    acc_row[j] = llaisys::utils::cast<float>(bias[j]);
                }
            } else {
                std::fill(acc_row.begin(), acc_row.end(), 0.0f);
            }

            for (size_t k = 0; k < K; k++) {
                const float in_val = llaisys::utils::cast<float>(in_row[k]);
                const T *w_row_t = weight_t + k * N;
                for (size_t j = 0; j < N; j++) {
                    acc_row[j] += in_val * llaisys::utils::cast<float>(w_row_t[j]);
                }
            }

            for (size_t j = 0; j < N; j++) {
                out_row[j] = llaisys::utils::cast<T>(acc_row[j]);
            }
        }
        return;
    }

    size_t rows_per = (M + nt - 1) / nt;
    std::vector<std::thread> workers;
    workers.reserve(nt);
    auto work = [&](size_t i_begin, size_t i_end) {
        std::vector<float> acc_row(N);
        for (size_t i = i_begin; i < i_end; i++) {
            const T *in_row = in + i * K;
            T *out_row = out + i * N;
            if (bias != nullptr) {
                for (size_t j = 0; j < N; j++) {
                    acc_row[j] = llaisys::utils::cast<float>(bias[j]);
                }
            } else {
                std::fill(acc_row.begin(), acc_row.end(), 0.0f);
            }

            for (size_t k = 0; k < K; k++) {
                const float in_val = llaisys::utils::cast<float>(in_row[k]);
                const T *w_row_t = weight_t + k * N;
                for (size_t j = 0; j < N; j++) {
                    acc_row[j] += in_val * llaisys::utils::cast<float>(w_row_t[j]);
                }
            }

            for (size_t j = 0; j < N; j++) {
                out_row[j] = llaisys::utils::cast<T>(acc_row[j]);
            }
        }
    };

    for (size_t t = 0; t < M; t += rows_per) {
        size_t i_begin = t;
        size_t i_end = std::min(i_begin + rows_per, M);
        if (i_begin >= i_end) {
            break;
        }
        workers.emplace_back(work, i_begin, i_end);
    }
    for (auto &t : workers) {
        t.join();
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t M, size_t N, size_t K, bool weight_transposed) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        if (weight_transposed) {
            return linear_t(reinterpret_cast<float *>(out),
                            reinterpret_cast<const float *>(in),
                            reinterpret_cast<const float *>(weight),
                            reinterpret_cast<const float *>(bias),
                            M, N, K);
        }
        return linear_nt(reinterpret_cast<float *>(out),
                         reinterpret_cast<const float *>(in),
                         reinterpret_cast<const float *>(weight),
                         reinterpret_cast<const float *>(bias),
                         M, N, K);
    case LLAISYS_DTYPE_F16:
        if (weight_transposed) {
            return linear_t(reinterpret_cast<llaisys::fp16_t *>(out),
                            reinterpret_cast<const llaisys::fp16_t *>(in),
                            reinterpret_cast<const llaisys::fp16_t *>(weight),
                            reinterpret_cast<const llaisys::fp16_t *>(bias),
                            M, N, K);
        }
        return linear_nt(reinterpret_cast<llaisys::fp16_t *>(out),
                         reinterpret_cast<const llaisys::fp16_t *>(in),
                         reinterpret_cast<const llaisys::fp16_t *>(weight),
                         reinterpret_cast<const llaisys::fp16_t *>(bias),
                         M, N, K);
    case LLAISYS_DTYPE_BF16:
        if (weight_transposed) {
            return linear_t(reinterpret_cast<llaisys::bf16_t *>(out),
                            reinterpret_cast<const llaisys::bf16_t *>(in),
                            reinterpret_cast<const llaisys::bf16_t *>(weight),
                            reinterpret_cast<const llaisys::bf16_t *>(bias),
                            M, N, K);
        }
        return linear_nt(reinterpret_cast<llaisys::bf16_t *>(out),
                         reinterpret_cast<const llaisys::bf16_t *>(in),
                         reinterpret_cast<const llaisys::bf16_t *>(weight),
                         reinterpret_cast<const llaisys::bf16_t *>(bias),
                         M, N, K);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
