"""SchedulerOutput → 张量与 AttentionMetadata。"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import torch

from nano_infer.debug import log as dlog
from nano_infer.debug import tensor_summary
from nano_infer.runner.attention_meta import AttentionMetadata

if TYPE_CHECKING:
    from nano_infer.config import EngineConfig
    from nano_infer.engine.scheduler.base import SchedulerOutput
    from nano_infer.engine.sequence import Sequence
    from nano_infer.models.base import BaseCausalLM


class ModelRunner:
    """
    准备 input_ids / positions / attn_metadata，调用 Model。
    kv_cache_pool 形状: [num_layers, 2, num_blocks, block_size, num_heads, head_dim] 等由具体 Model 约定；
    MVP 下 Dummy 模型可使用占位张量。
    """

    def __init__(self, model: "BaseCausalLM", config: "EngineConfig"):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype) if hasattr(torch, config.dtype) else torch.float16
        self.kv_cache_pool: torch.Tensor | None = None

    def _max_blocks_per_seq(self) -> int:
        return (self.config.max_model_len + self.config.block_size - 1) // self.config.block_size

    def prepare_inputs(
        self, sched_out: "SchedulerOutput"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ids: List[int] = []
        pos: List[int] = []
        for seq, n in zip(sched_out.sequences, sched_out.num_tokens_per_seq):
            if sched_out.is_prefill:
                start = seq.num_computed_tokens
                for i in range(n):
                    ids.append(seq.prompt_ids[start + i])
                    pos.append(start + i)
            else:
                # decode: 单 token = 上一拍生成的最后一个（首拍 decode 用 output 最后一项）
                if seq.output_ids:
                    tid = seq.output_ids[-1]
                else:
                    tid = seq.prompt_ids[-1]
                ids.append(tid)
                pos.append(seq.num_computed_tokens)
        t_ids = torch.tensor(ids, device=self.device, dtype=torch.long)
        t_pos = torch.tensor(pos, device=self.device, dtype=torch.long)
        dlog("runner", f"prepare_inputs: ids={t_ids.tolist()[:8]}{'...' if len(ids) > 8 else ''} "
             f"pos={t_pos.tolist()[:8]}{'...' if len(pos) > 8 else ''} total={len(ids)}")
        return t_ids, t_pos

    def prepare_metadata(self, sched_out: "SchedulerOutput") -> AttentionMetadata:
        seqs = sched_out.sequences
        nseq = len(seqs)
        max_b = self._max_blocks_per_seq()
        block_tables = torch.zeros(nseq, max_b, device=self.device, dtype=torch.long)
        context_lens = []
        seq_lens_after = []
        slots: List[int] = []
        block_size = self.config.block_size

        for i, (seq, n) in enumerate(zip(seqs, sched_out.num_tokens_per_seq)):
            bt = seq.block_table[:max_b]
            if len(bt) < max_b:
                row = seq.block_table + [0] * (max_b - len(bt))
            else:
                row = bt[:max_b]
            block_tables[i, :].copy_(torch.tensor(row, device=self.device))

            ctx = seq.num_computed_tokens
            context_lens.append(ctx)
            if sched_out.is_prefill:
                after = ctx + n
            else:
                after = ctx + 1
            seq_lens_after.append(after)

            for t in range(n):
                global_tok = ctx + t
                block_idx = global_tok // block_size
                off = global_tok % block_size
                phys = seq.block_table[block_idx] if block_idx < len(seq.block_table) else 0
                slot = phys * block_size + off
                slots.append(slot)

        total = sum(sched_out.num_tokens_per_seq)
        max_q = max(sched_out.num_tokens_per_seq)
        max_k = max(seq_lens_after)

        meta = AttentionMetadata(
            num_seqs=nseq,
            total_tokens=total,
            seq_lens=torch.tensor(seq_lens_after, device=self.device, dtype=torch.int32),
            context_lens=torch.tensor(context_lens, device=self.device, dtype=torch.int32),
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            block_tables=block_tables,
            slot_mapping=torch.tensor(slots, device=self.device, dtype=torch.long),
            block_size=block_size,
            device=self.device,
            dtype=self.dtype,
        )
        dlog("runner", f"metadata: num_seqs={nseq} total_tokens={total} "
             f"seq_lens={seq_lens_after} context_lens={context_lens} "
             f"max_q={max_q} max_k={max_k} slots={slots[:8]}{'...' if len(slots) > 8 else ''}")
        return meta

    @torch.inference_mode()
    def execute_model(self, sched_out: "SchedulerOutput") -> torch.Tensor:
        attn_metadata = self.prepare_metadata(sched_out)
        input_ids, positions = self.prepare_inputs(sched_out)
        if self.kv_cache_pool is None:
            nb = self.config.num_gpu_blocks
            n_layer = self.config.num_hidden_layers
            n_kv_heads = self.config.num_kv_heads
            hd = self.config.head_dim
            self.kv_cache_pool = torch.zeros(
                n_layer, 2, nb, self.config.block_size, n_kv_heads, hd,
                device=self.device,
                dtype=self.dtype,
            )
            dlog("runner", f"kv_cache_pool allocated: shape={list(self.kv_cache_pool.shape)} "
                 f"n_q_heads={self.config.num_attention_heads} n_kv_heads={n_kv_heads} "
                 f"dtype={self.dtype} device={self.device}")
        logits = self.model(
            input_ids=input_ids,
            positions=positions,
            kv_cache=self.kv_cache_pool,
            attn_metadata=attn_metadata,
        )
        dlog("runner", f"model output: {tensor_summary(logits, 'logits')}")
        return logits
