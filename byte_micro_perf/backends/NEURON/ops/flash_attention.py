import sys
import pathlib
from functools import partial
import torch

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[3])
)

from core.ops.llm_ops import FlashAttentionOp
from core.utils import OpTensorInfo, calc_tensor_size

OP_MAPPING = {}


try:
    from neuronxcc.nki.kernels.attention import flash_fwd, FlashConfig
    import torch_neuronx

    class NKIFlashAttentionOp(FlashAttentionOp):
        """Flash Attention using NKI flash_fwd kernel on AWS Neuron.

        NKI flash_fwd expects:
          q: (bs, n_heads, d, seq_q)
          k: (bs, nk_heads, d, seq_k)
          v: (bs, nv_heads, d, seq_v)  when should_transpose_v=True
          o: (bs, n_heads, seq_q, d)

        The benchmark FlashAttentionOp provides:
          q: (num_tokens, q_head_num, head_dim) -- for prefill with batch_size=1
          k_cache/v_cache: (batch_size, kv_head_num, max_kv_len, head_dim)
        """

        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

            if self.attn_mode == "prefill":
                self.prefill_init()
            elif self.attn_mode == "decode":
                raise NotImplementedError(
                    "NKI flash_fwd does not support decode mode (variable-length KV cache). "
                    "Use prefill mode for NKI flash attention benchmarking."
                )

        def prefill_init(self):
            if not (
                self.dtype == "bfloat16"
                and self.compute_dtype == "bfloat16"
                and self.cache_dtype == "bfloat16"
            ):
                raise ValueError(
                    "NKI FlashAttention only supports bfloat16 for dtype, compute_dtype and cache_dtype"
                )

            if not (self.cache_type == "linear"):
                raise ValueError(
                    "NKI FlashAttention only supports linear cache_type"
                )

            if not (self.batch_size == 1 and self.cache_lens[0] == 0):
                raise ValueError(
                    "NKI FlashAttention only supports prefill with batch_size=1 and cache_len=0"
                )

            # Seq length must be divisible by seq_tile_size (default 2048)
            seq_len = self.num_tokens
            if seq_len >= 2048:
                self._seq_tile_size = 2048
            elif seq_len >= 1024:
                self._seq_tile_size = seq_len
            else:
                self._seq_tile_size = seq_len

            self._flash_config = FlashConfig(
                seq_tile_size=self._seq_tile_size,
                training=False,
                should_transpose_v=True,
            )

            self._run_func = self.prefill_run

        def prefill_run(self, tensor_mapping):
            q = tensor_mapping["q"]
            k_cache = tensor_mapping["k_cache"]
            v_cache = tensor_mapping["v_cache"]

            # Reshape from benchmark format to NKI format
            # q: (num_tokens, q_head_num, head_dim) -> (1, q_head_num, head_dim, seq_q)
            q_nki = q.view(
                self.batch_size, self.num_tokens, self.q_head_num, self.head_dim
            ).permute(0, 2, 3, 1).contiguous()

            # k_cache: (bs, kv_head_num, max_kv_len, head_dim) -> (bs, kv_head_num, head_dim, seq_k)
            k_nki = k_cache[:, :, : self.num_tokens, :].permute(0, 1, 3, 2).contiguous()

            # v_cache: (bs, kv_head_num, max_kv_len, head_dim) -> (bs, kv_head_num, head_dim, seq_v)
            # should_transpose_v=True means v is (bs, nv_heads, d, seq_v)
            v_nki = v_cache[:, :, : self.num_tokens, :].permute(0, 1, 3, 2).contiguous()

            # Allocate output: (bs, n_heads, seq_q, head_dim)
            o_nki = torch.empty(
                self.batch_size,
                self.q_head_num,
                self.num_tokens,
                self.head_dim,
                dtype=self.torch_dtype,
                device=q.device,
            )

            flash_fwd[self.batch_size, self.q_head_num](
                q_nki,
                k_nki,
                v_nki,
                use_causal_mask=self.is_causal,
                mixed_precision=True,
                softmax_scale=self.softmax_scale,
                config=self._flash_config,
                o=o_nki,
            )

            return o_nki

    OP_MAPPING["nki"] = NKIFlashAttentionOp

except ImportError:
    pass
