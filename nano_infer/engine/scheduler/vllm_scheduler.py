"""FCFS + 连续批处理（vLLM 风格，MVP）。"""

from __future__ import annotations

from typing import List

from nano_infer.engine.scheduler.base import SchedulerBase, SchedulerOutput, _ensure_blocks
from nano_infer.engine.sequence import SequenceStatus


class VLLMScheduler(SchedulerBase):
    """
    先到先得；单步内仅 prefill 批或仅 decode 批（MVP 简化）。
    prefill 按 max_num_batched_tokens 切块。
    """

    def schedule(self, waiting, running, block_manager):
        cfg = self.config
        ordered: List = []
        seen = set()
        for s in running:
            if s.seq_id not in seen and s.status != SequenceStatus.FINISHED:
                ordered.append(s)
                seen.add(s.seq_id)
        for s in waiting:
            if s.seq_id not in seen:
                ordered.append(s)
                seen.add(s.seq_id)

        active = [s for s in ordered if s.status != SequenceStatus.FINISHED]
        if not active:
            return None

        prefill_candidates = [s for s in active if not s.is_prefill_done()]
        if prefill_candidates:
            batch: List = []
            num_tokens_per_seq: List[int] = []
            total_tok = 0
            for seq in prefill_candidates:
                if len(batch) >= cfg.max_num_seqs:
                    break
                if seq.status == SequenceStatus.WAITING:
                    seq.status = SequenceStatus.PREFILL
                rem = seq.prefill_remaining()
                take = min(rem, cfg.max_num_batched_tokens - total_tok)
                if take <= 0:
                    break
                after = seq.num_computed_tokens + take
                if not _ensure_blocks(seq, block_manager, after):
                    break
                batch.append(seq)
                num_tokens_per_seq.append(take)
                total_tok += take
            if batch:
                return SchedulerOutput(batch, num_tokens_per_seq, is_prefill=True)

        decode_candidates = [s for s in active if s.is_prefill_done()]
        batch = []
        num_tokens_per_seq = []
        total_tok = 0
        for seq in decode_candidates:
            if len(batch) >= cfg.max_num_seqs:
                break
            if seq.status == SequenceStatus.PREFILL:
                seq.status = SequenceStatus.DECODE
            if seq.status == SequenceStatus.WAITING:
                continue
            after = seq.total_len()
            if not _ensure_blocks(seq, block_manager, after):
                break
            batch.append(seq)
            num_tokens_per_seq.append(1)
            total_tok += 1
            if total_tok >= cfg.max_num_batched_tokens:
                break

        if not batch:
            return None
        return SchedulerOutput(batch, num_tokens_per_seq, is_prefill=False)
