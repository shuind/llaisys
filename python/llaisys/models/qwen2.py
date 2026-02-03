from typing import Sequence
import json
from ctypes import c_int, c_int64, c_size_t, c_void_p, byref
from pathlib import Path

import torch
from safetensors import safe_open

from ..libllaisys import LIB_LLAISYS, DeviceType, DataType, llaisysDeviceType_t
from ..libllaisys.models import LlaisysQwen2Meta


_DTYPE_MAP = {
    "bfloat16": DataType.BF16,
    "float16": DataType.F16,
    "float32": DataType.F32,
}


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        config_path = model_path / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        dtype = _DTYPE_MAP.get(cfg.get("torch_dtype", "bfloat16"), DataType.BF16)
        meta = LlaisysQwen2Meta(
            dtype,
            cfg["num_hidden_layers"],
            cfg["hidden_size"],
            cfg["num_attention_heads"],
            cfg["num_key_value_heads"],
            cfg["hidden_size"] // cfg["num_attention_heads"],
            cfg["intermediate_size"],
            cfg["max_position_embeddings"],
            cfg["vocab_size"],
            float(cfg["rms_norm_eps"]),
            float(cfg["rope_theta"]),
            int(cfg["eos_token_id"]),
        )

        device_ids = (c_int * 1)(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(meta), llaisysDeviceType_t(device), device_ids, 1
        )
        self._weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model).contents
        self._meta = meta
        self._device = device

        self._load_weights(model_path)

    def __del__(self):
        if hasattr(self, "_model") and self._model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None

    def _load_tensor(self, handle, torch_tensor):
        tensor = torch_tensor.contiguous()
        LIB_LLAISYS.tensorLoad(handle, c_void_p(tensor.data_ptr()))

    def _load_by_name(self, name, torch_tensor):
        w = self._weights
        if name == "model.embed_tokens.weight":
            self._load_tensor(w.in_embed, torch_tensor)
            return True
        if name == "lm_head.weight":
            self._load_tensor(w.out_embed, torch_tensor)
            return True
        if name == "model.norm.weight":
            self._load_tensor(w.out_norm_w, torch_tensor)
            return True
        if not name.startswith("model.layers."):
            return False

        parts = name.split(".")
        layer = int(parts[2])
        sub = ".".join(parts[3:])

        if sub == "input_layernorm.weight":
            self._load_tensor(w.attn_norm_w[layer], torch_tensor)
        elif sub == "post_attention_layernorm.weight":
            self._load_tensor(w.mlp_norm_w[layer], torch_tensor)
        elif sub == "self_attn.q_proj.weight":
            self._load_tensor(w.attn_q_w[layer], torch_tensor)
        elif sub == "self_attn.q_proj.bias":
            self._load_tensor(w.attn_q_b[layer], torch_tensor)
        elif sub == "self_attn.k_proj.weight":
            self._load_tensor(w.attn_k_w[layer], torch_tensor)
        elif sub == "self_attn.k_proj.bias":
            self._load_tensor(w.attn_k_b[layer], torch_tensor)
        elif sub == "self_attn.v_proj.weight":
            self._load_tensor(w.attn_v_w[layer], torch_tensor)
        elif sub == "self_attn.v_proj.bias":
            self._load_tensor(w.attn_v_b[layer], torch_tensor)
        elif sub == "self_attn.o_proj.weight":
            self._load_tensor(w.attn_o_w[layer], torch_tensor)
        elif sub == "mlp.gate_proj.weight":
            self._load_tensor(w.mlp_gate_w[layer], torch_tensor)
        elif sub == "mlp.up_proj.weight":
            self._load_tensor(w.mlp_up_w[layer], torch_tensor)
        elif sub == "mlp.down_proj.weight":
            self._load_tensor(w.mlp_down_w[layer], torch_tensor)
        else:
            return False
        return True

    def _load_weights(self, model_path: Path):
        for file in sorted(model_path.glob("*.safetensors")):
            with safe_open(file, framework="torch", device="cpu") as data_:
                for name_ in data_.keys():
                    tensor = data_.get_tensor(name_)
                    self._load_by_name(name_, tensor)

    def _infer(self, token_ids):
        arr = (c_int64 * len(token_ids))(*token_ids)
        return int(
            LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model, arr, c_size_t(len(token_ids))
            )
        )

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        if max_new_tokens is None:
            max_new_tokens = 128
        if max_new_tokens <= 0:
            return list(inputs)

        tokens = list(inputs)
        if not tokens:
            return []

        next_id = self._infer(tokens)
        tokens.append(next_id)
        for _ in range(max_new_tokens - 1):
            if next_id == self._meta.end_token:
                break
            next_id = self._infer([tokens[-1]])
            tokens.append(next_id)

        return tokens
