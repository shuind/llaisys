#include "llaisys/models/qwen2.h"
#include "llaisys_tensor.hpp"

#include "../core/llaisys_core.hpp"
#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "../tensor/tensor.hpp"
#include "../utils.hpp"

#include <cmath>
#include <cstring>
#include <vector>

namespace {
llaisysTensor_t make_tensor(const std::vector<size_t> &shape,
                            llaisysDataType_t dtype,
                            llaisysDeviceType_t device,
                            int device_id) {
    return new LlaisysTensor{llaisys::Tensor::create(shape, dtype, device, device_id)};
}

llaisys::tensor_t unwrap(llaisysTensor_t tensor) {
    return tensor->tensor;
}

llaisys::tensor_t concat_cache(const llaisys::tensor_t &old_cache,
                               const llaisys::tensor_t &new_data,
                               size_t total_len) {
    std::vector<size_t> shape = {total_len, old_cache->shape()[1], old_cache->shape()[2]};
    auto out = llaisys::Tensor::create(shape, old_cache->dtype(), old_cache->deviceType(), old_cache->deviceId());
    size_t elem_size = old_cache->elementSize();
    size_t old_bytes = old_cache->numel() * elem_size;
    size_t new_bytes = new_data->numel() * elem_size;

    if (old_cache->deviceType() == LLAISYS_DEVICE_CPU) {
        std::memcpy(out->data(), old_cache->data(), old_bytes);
        std::memcpy(out->data() + old_bytes, new_data->data(), new_bytes);
        return out;
    }

    llaisys::core::context().setDevice(old_cache->deviceType(), old_cache->deviceId());
    auto api = llaisys::core::context().runtime().api();
    api->memcpy_sync(out->data(), old_cache->data(), old_bytes, LLAISYS_MEMCPY_D2D);
    api->memcpy_sync(out->data() + old_bytes, new_data->data(), new_bytes, LLAISYS_MEMCPY_D2D);
    return out;
}
} // namespace

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    llaisysDeviceType_t device;
    int device_id;
    LlaisysQwen2Weights weights;
    std::vector<llaisys::tensor_t> kcache;
    std::vector<llaisys::tensor_t> vcache;
    size_t cache_len = 0;
};

__C {
    LlaisysQwen2Model *llaisysQwen2ModelCreate(
        const LlaisysQwen2Meta *meta,
        llaisysDeviceType_t device,
        int *device_ids,
        int ndevice) {
        try {
            if (!meta) {
                return nullptr;
            }
            auto model = new LlaisysQwen2Model();
            model->meta = *meta;
            model->device = device;
            model->device_id = (device_ids && ndevice > 0) ? device_ids[0] : 0;

        size_t nlayer = meta->nlayer;
        size_t hs = meta->hs;
        size_t nh = meta->nh;
        size_t nkvh = meta->nkvh;
        size_t dh = meta->dh;
        size_t di = meta->di;
        size_t voc = meta->voc;

        model->weights.in_embed = make_tensor({voc, hs}, meta->dtype, device, model->device_id);
        model->weights.out_embed = make_tensor({hs, voc}, meta->dtype, device, model->device_id);
        model->weights.out_norm_w = make_tensor({hs}, meta->dtype, device, model->device_id);

        model->weights.attn_norm_w = new llaisysTensor_t[nlayer];
        model->weights.attn_q_w = new llaisysTensor_t[nlayer];
        model->weights.attn_q_b = new llaisysTensor_t[nlayer];
        model->weights.attn_k_w = new llaisysTensor_t[nlayer];
        model->weights.attn_k_b = new llaisysTensor_t[nlayer];
        model->weights.attn_v_w = new llaisysTensor_t[nlayer];
        model->weights.attn_v_b = new llaisysTensor_t[nlayer];
        model->weights.attn_o_w = new llaisysTensor_t[nlayer];
        model->weights.mlp_norm_w = new llaisysTensor_t[nlayer];
        model->weights.mlp_gate_w = new llaisysTensor_t[nlayer];
        model->weights.mlp_up_w = new llaisysTensor_t[nlayer];
        model->weights.mlp_down_w = new llaisysTensor_t[nlayer];

        size_t kv_out = nkvh * dh;
        for (size_t i = 0; i < nlayer; i++) {
            model->weights.attn_norm_w[i] = make_tensor({hs}, meta->dtype, device, model->device_id);
            model->weights.attn_q_w[i] = make_tensor({hs, hs}, meta->dtype, device, model->device_id);
            model->weights.attn_q_b[i] = make_tensor({hs}, meta->dtype, device, model->device_id);
            model->weights.attn_k_w[i] = make_tensor({hs, kv_out}, meta->dtype, device, model->device_id);
            model->weights.attn_k_b[i] = make_tensor({kv_out}, meta->dtype, device, model->device_id);
            model->weights.attn_v_w[i] = make_tensor({hs, kv_out}, meta->dtype, device, model->device_id);
            model->weights.attn_v_b[i] = make_tensor({kv_out}, meta->dtype, device, model->device_id);
            model->weights.attn_o_w[i] = make_tensor({hs, hs}, meta->dtype, device, model->device_id);
            model->weights.mlp_norm_w[i] = make_tensor({hs}, meta->dtype, device, model->device_id);
            model->weights.mlp_gate_w[i] = make_tensor({hs, di}, meta->dtype, device, model->device_id);
            model->weights.mlp_up_w[i] = make_tensor({hs, di}, meta->dtype, device, model->device_id);
            model->weights.mlp_down_w[i] = make_tensor({di, hs}, meta->dtype, device, model->device_id);
        }

        model->kcache.resize(nlayer);
        model->vcache.resize(nlayer);
        model->cache_len = 0;

            return model;
        } catch (...) {
            return nullptr;
        }
    }

    void llaisysQwen2ModelDestroy(LlaisysQwen2Model *model) {
        if (!model) {
            return;
        }

        auto free_tensor = [](llaisysTensor_t t) {
            delete t;
        };

        free_tensor(model->weights.in_embed);
        free_tensor(model->weights.out_embed);
        free_tensor(model->weights.out_norm_w);

        size_t nlayer = model->meta.nlayer;
        for (size_t i = 0; i < nlayer; i++) {
            free_tensor(model->weights.attn_norm_w[i]);
            free_tensor(model->weights.attn_q_w[i]);
            free_tensor(model->weights.attn_q_b[i]);
            free_tensor(model->weights.attn_k_w[i]);
            free_tensor(model->weights.attn_k_b[i]);
            free_tensor(model->weights.attn_v_w[i]);
            free_tensor(model->weights.attn_v_b[i]);
            free_tensor(model->weights.attn_o_w[i]);
            free_tensor(model->weights.mlp_norm_w[i]);
            free_tensor(model->weights.mlp_gate_w[i]);
            free_tensor(model->weights.mlp_up_w[i]);
            free_tensor(model->weights.mlp_down_w[i]);
        }

        delete[] model->weights.attn_norm_w;
        delete[] model->weights.attn_q_w;
        delete[] model->weights.attn_q_b;
        delete[] model->weights.attn_k_w;
        delete[] model->weights.attn_k_b;
        delete[] model->weights.attn_v_w;
        delete[] model->weights.attn_v_b;
        delete[] model->weights.attn_o_w;
        delete[] model->weights.mlp_norm_w;
        delete[] model->weights.mlp_gate_w;
        delete[] model->weights.mlp_up_w;
        delete[] model->weights.mlp_down_w;

        delete model;
    }

    LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model *model) {
        return model ? &model->weights : nullptr;
    }

    int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
        try {
            if (!model || !token_ids || ntoken == 0) {
                return -1;
            }

        auto &meta = model->meta;
        size_t L = ntoken;
        size_t H = meta.nh;
        size_t HKV = meta.nkvh;
        size_t D = meta.dh;
        size_t hs = meta.hs;
        size_t di = meta.di;
        size_t voc = meta.voc;
        float scale = 1.0f / std::sqrt(static_cast<float>(D));

            if (model->device != LLAISYS_DEVICE_CPU) {
                return -1;
            }

        auto token_tensor = llaisys::Tensor::create({L}, LLAISYS_DTYPE_I64, model->device, model->device_id);
        token_tensor->load(token_ids);

        auto x = llaisys::Tensor::create({L, hs}, meta.dtype, model->device, model->device_id);
        llaisys::ops::embedding(x, token_tensor, unwrap(model->weights.in_embed));

        std::vector<int64_t> pos_ids_vec(L);
        size_t start_pos = model->cache_len;
        for (size_t i = 0; i < L; i++) {
            pos_ids_vec[i] = static_cast<int64_t>(start_pos + i);
        }
        auto pos_ids = llaisys::Tensor::create({L}, LLAISYS_DTYPE_I64, model->device, model->device_id);
        pos_ids->load(pos_ids_vec.data());

        for (size_t layer = 0; layer < meta.nlayer; layer++) {
            auto norm1 = llaisys::Tensor::create({L, hs}, meta.dtype, model->device, model->device_id);
            llaisys::ops::rms_norm(norm1, x, unwrap(model->weights.attn_norm_w[layer]), meta.epsilon);

            auto q2d = llaisys::Tensor::create({L, hs}, meta.dtype, model->device, model->device_id);
            auto k2d = llaisys::Tensor::create({L, HKV * D}, meta.dtype, model->device, model->device_id);
            auto v2d = llaisys::Tensor::create({L, HKV * D}, meta.dtype, model->device, model->device_id);
            llaisys::ops::linear_transposed(q2d, norm1, unwrap(model->weights.attn_q_w[layer]),
                                            unwrap(model->weights.attn_q_b[layer]));
            llaisys::ops::linear_transposed(k2d, norm1, unwrap(model->weights.attn_k_w[layer]),
                                            unwrap(model->weights.attn_k_b[layer]));
            llaisys::ops::linear_transposed(v2d, norm1, unwrap(model->weights.attn_v_w[layer]),
                                            unwrap(model->weights.attn_v_b[layer]));

            auto q = q2d->view({L, H, D});
            auto k = k2d->view({L, HKV, D});
            auto v = v2d->view({L, HKV, D});

            auto q_rope = llaisys::Tensor::create({L, H, D}, meta.dtype, model->device, model->device_id);
            auto k_rope = llaisys::Tensor::create({L, HKV, D}, meta.dtype, model->device, model->device_id);
            llaisys::ops::rope(q_rope, q, pos_ids, meta.theta);
            llaisys::ops::rope(k_rope, k, pos_ids, meta.theta);

            llaisys::tensor_t k_all = k_rope;
            llaisys::tensor_t v_all = v;
            if (model->cache_len > 0) {
                k_all = concat_cache(model->kcache[layer], k_rope, model->cache_len + L);
                v_all = concat_cache(model->vcache[layer], v, model->cache_len + L);
            }
            model->kcache[layer] = k_all;
            model->vcache[layer] = v_all;

            auto attn_val = llaisys::Tensor::create({L, H, D}, meta.dtype, model->device, model->device_id);
            llaisys::ops::self_attention(attn_val, q_rope, k_all, v_all, scale);

            auto attn_val2d = attn_val->view({L, hs});
            auto attn_out = llaisys::Tensor::create({L, hs}, meta.dtype, model->device, model->device_id);
            llaisys::ops::linear_transposed(attn_out, attn_val2d, unwrap(model->weights.attn_o_w[layer]), nullptr);

            auto attn_res = llaisys::Tensor::create({L, hs}, meta.dtype, model->device, model->device_id);
            llaisys::ops::add(attn_res, x, attn_out);

            auto norm2 = llaisys::Tensor::create({L, hs}, meta.dtype, model->device, model->device_id);
            llaisys::ops::rms_norm(norm2, attn_res, unwrap(model->weights.mlp_norm_w[layer]), meta.epsilon);

            auto gate = llaisys::Tensor::create({L, di}, meta.dtype, model->device, model->device_id);
            auto up = llaisys::Tensor::create({L, di}, meta.dtype, model->device, model->device_id);
            llaisys::ops::linear_transposed(gate, norm2, unwrap(model->weights.mlp_gate_w[layer]), nullptr);
            llaisys::ops::linear_transposed(up, norm2, unwrap(model->weights.mlp_up_w[layer]), nullptr);

            auto swiglu_out = llaisys::Tensor::create({L, di}, meta.dtype, model->device, model->device_id);
            llaisys::ops::swiglu(swiglu_out, gate, up);

            auto mlp_out = llaisys::Tensor::create({L, hs}, meta.dtype, model->device, model->device_id);
            llaisys::ops::linear_transposed(mlp_out, swiglu_out, unwrap(model->weights.mlp_down_w[layer]), nullptr);

            auto mlp_res = llaisys::Tensor::create({L, hs}, meta.dtype, model->device, model->device_id);
            llaisys::ops::add(mlp_res, attn_res, mlp_out);

            x = mlp_res;
        }

        model->cache_len += L;

        auto out_norm = llaisys::Tensor::create({L, hs}, meta.dtype, model->device, model->device_id);
        llaisys::ops::rms_norm(out_norm, x, unwrap(model->weights.out_norm_w), meta.epsilon);

        auto last = out_norm->slice(0, L - 1, L);
        auto logits = llaisys::Tensor::create({1, voc}, meta.dtype, model->device, model->device_id);
        llaisys::ops::linear_transposed(logits, last, unwrap(model->weights.out_embed), nullptr);

        auto logits_1d = logits->view({voc});
        auto max_idx = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, model->device, model->device_id);
        auto max_val = llaisys::Tensor::create({1}, meta.dtype, model->device, model->device_id);
        llaisys::ops::argmax(max_idx, max_val, logits_1d);

        auto idx_ptr = reinterpret_cast<int64_t *>(max_idx->data());
            return idx_ptr[0];
        } catch (...) {
            return -1;
        }
    }
}
