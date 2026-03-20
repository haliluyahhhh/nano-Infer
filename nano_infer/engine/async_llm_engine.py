"""主事件循环：调度 → Runner → 采样 → 更新序列。"""

from __future__ import annotations

import asyncio
import itertools
from typing import AsyncIterator, List, Optional, Tuple

import torch

from nano_infer.config import EngineConfig
from nano_infer.debug import log as dlog
from nano_infer.engine.memory.block_manager import BlockManager
from nano_infer.engine.scheduler.base import SchedulerOutput
from nano_infer.engine.sequence import Sequence, SequenceStatus
from nano_infer.runner.model_runner import ModelRunner


class AsyncLLMEngine:
    def __init__(self, runner: ModelRunner, scheduler, block_manager: BlockManager):
        self.runner = runner
        self.scheduler = scheduler
        self.block_manager = block_manager
        self._waiting: List[Sequence] = []
        self._running: List[Sequence] = []
        self._lock = asyncio.Lock()
        self._id_gen = itertools.count(1)

    def _schedule(self) -> Optional[SchedulerOutput]:
        out = self.scheduler.schedule(self._waiting, self._running, self.block_manager)
        if out:
            for s in out.sequences:
                if s in self._waiting:
                    self._waiting.remove(s)
                if s not in self._running:
                    self._running.append(s)
        return out

    def _logits_indices_for_last_tokens(self, sched_out: SchedulerOutput) -> List[int]:
        acc = 0
        idxs = []
        for n in sched_out.num_tokens_per_seq:
            idxs.append(acc + n - 1)
            acc += n
        return idxs

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        token_ids: list[int],
        penalty: float,
    ) -> torch.Tensor:
        """对已出现过的 token 施加重复惩罚（>1 抑制，<1 鼓励）。"""
        if penalty == 1.0 or not token_ids:
            return logits
        seen = torch.tensor(list(set(token_ids)), dtype=torch.long, device=logits.device)
        scores = logits[seen]
        # 正 logit → 除以 penalty；负 logit → 乘以 penalty（与 HF 实现一致）
        scores = torch.where(scores > 0, scores / penalty, scores * penalty)
        logits = logits.clone()
        logits[seen] = scores
        return logits

    @staticmethod
    def _apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 0 or k >= logits.shape[-1]:
            return logits
        top_k_vals = torch.topk(logits, k).values
        logits = logits.clone()
        logits[logits < top_k_vals[..., -1]] = float("-inf")
        return logits

    @staticmethod
    def _apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
        if p >= 1.0:
            return logits
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cum_probs - torch.softmax(sorted_logits, dim=-1) >= p
        sorted_logits[mask] = float("-inf")
        logits = logits.clone()
        logits.scatter_(0, sorted_idx, sorted_logits)
        return logits

    def _sample(self, logits_row: torch.Tensor, seq: "Sequence") -> int:
        temperature = seq.temperature
        rep_penalty = seq.repetition_penalty
        top_k = seq.top_k
        top_p = seq.top_p

        logits = logits_row
        all_ids = seq.prompt_ids + seq.output_ids
        logits = self._apply_repetition_penalty(logits, all_ids, rep_penalty)

        if temperature <= 0:
            return int(logits.argmax().item())

        logits = logits / temperature
        logits = self._apply_top_k(logits, top_k)
        logits = self._apply_top_p(logits, top_p)
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1).item())

    def step(self) -> bool:
        """
        执行一步前向。若本步无工作返回 False。
        返回 True 表示可能仍有未完成序列。
        """
        sched_out = self._schedule()
        if sched_out is None:
            dlog("engine", "step: no work scheduled",
                 waiting=len(self._waiting), running=len(self._running))
            return any(s.status != SequenceStatus.FINISHED for s in self._running) or bool(
                self._waiting
            )

        dlog("engine", f"step: {'PREFILL' if sched_out.is_prefill else 'DECODE'} "
             f"seqs={len(sched_out.sequences)} tokens={sched_out.num_tokens_per_seq}")
        logits = self.runner.execute_model(sched_out)
        last_idx = self._logits_indices_for_last_tokens(sched_out)

        for i, seq in enumerate(sched_out.sequences):
            li = last_idx[i]
            row = logits[li]

            if sched_out.is_prefill:
                n = sched_out.num_tokens_per_seq[i]
                seq.num_computed_tokens += n
                if seq.is_prefill_done():
                    tid = self._sample(row, seq)
                    seq.output_ids.append(tid)
                    dlog("engine", f"  seq[{seq.seq_id}] prefill done, first token={tid}, "
                         f"logits_top5={torch.topk(row, 5).indices.tolist()}")
                    if seq.eos_token_id is not None and tid == seq.eos_token_id:
                        seq.status = SequenceStatus.FINISHED
                        self.block_manager.free_sequence(seq.seq_id)
                        self._running = [s for s in self._running if s.seq_id != seq.seq_id]
                    elif len(seq.output_ids) >= seq.max_tokens:
                        seq.status = SequenceStatus.FINISHED
                        self.block_manager.free_sequence(seq.seq_id)
                        self._running = [s for s in self._running if s.seq_id != seq.seq_id]
            else:
                tid = self._sample(row, seq)
                seq.output_ids.append(tid)
                seq.num_computed_tokens += 1
                dlog("engine", f"  seq[{seq.seq_id}] decode token={tid}, "
                     f"pos={seq.num_computed_tokens}, out_len={len(seq.output_ids)}")
                if seq.eos_token_id is not None and tid == seq.eos_token_id:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.free_sequence(seq.seq_id)
                    self._running = [s for s in self._running if s.seq_id != seq.seq_id]
                elif len(seq.output_ids) >= seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.free_sequence(seq.seq_id)
                    self._running = [s for s in self._running if s.seq_id != seq.seq_id]

        return True

    def add_sequence(self, seq: Sequence) -> None:
        self._waiting.append(seq)

    def create_sequence(
        self,
        prompt_ids: List[int],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> Sequence:
        cfg = self.runner.config
        return Sequence(
            seq_id=next(self._id_gen),
            prompt_ids=prompt_ids,
            max_tokens=max_tokens or cfg.max_tokens,
            temperature=temperature if temperature is not None else cfg.temperature,
            top_p=top_p if top_p is not None else cfg.top_p,
            top_k=top_k if top_k is not None else cfg.top_k,
            repetition_penalty=repetition_penalty if repetition_penalty is not None else cfg.repetition_penalty,
            eos_token_id=eos_token_id,
        )

    async def generate_tokens(
        self,
        prompt_ids: List[int],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> AsyncIterator[int]:
        seq = self.create_sequence(
            prompt_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_token_id,
        )
        self.add_sequence(seq)
        loop = asyncio.get_event_loop()
        while seq.status != SequenceStatus.FINISHED:
            prev_len = len(seq.output_ids)
            await loop.run_in_executor(None, self.step)
            for j in range(prev_len, len(seq.output_ids)):
                yield seq.output_ids[j]
            if not self._running and not self._waiting and seq.status != SequenceStatus.FINISHED:
                break
            await asyncio.sleep(0)


def build_engine(config: EngineConfig) -> Tuple[AsyncLLMEngine, ModelRunner]:
    import nano_infer.models  # noqa: F401 — 注册 dummy/llama3/qwen2/qwen3
    from nano_infer.config import load_model_config_from_path
    from nano_infer.models.registry import get_model_class
    from nano_infer.models.weight_loader import load_hf_weights

    from nano_infer.models.model_config import detect_model_type

    cfg = config
    weights: dict | None = None
    if cfg.model_path:
        mc = load_model_config_from_path(cfg.model_path)
        cfg.merge_from_model_config(mc)
        detected = detect_model_type(cfg.model_path)
        cfg.model_name = detected
        # 先加载权重再建模型：Qwen2 等 config 常不写 attention_bias，但 checkpoint 含 QKV bias；
        # 若此处不检测，Linear(bias=False) 会丢弃全部 bias，输出接近随机。
        weights = load_hf_weights(cfg.model_path, device=cfg.device)
        has_qkv_bias = any(k.endswith("self_attn.q_proj.bias") for k in weights)
        if has_qkv_bias:
            cfg.attention_bias = True
        dlog("config", f"model_path={cfg.model_path} detected_type={detected} "
             f"final_model_name={cfg.model_name}")
        dlog("config", f"  hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} "
             f"heads={cfg.num_attention_heads} kv_heads={cfg.num_key_value_heads} "
             f"head_dim={cfg.head_dim} head_dim_override={cfg.head_dim_override} "
             f"vocab={cfg.vocab_size} intermediate={cfg.intermediate_size} "
             f"rope_theta={cfg.rope_theta} "
             f"attention_bias={cfg.attention_bias} (from_weights={has_qkv_bias}) "
             f"tie_word_embeddings={cfg.tie_word_embeddings}")

    cls = get_model_class(cfg.model_name)
    model = cls(cfg)
    device = torch.device(cfg.device)
    dtype = getattr(torch, cfg.dtype) if hasattr(torch, cfg.dtype) else torch.float32
    dlog("config", f"device={device} dtype={dtype}")
    model.to(device=device, dtype=dtype)

    if cfg.model_path and weights is not None:
        model.load_weights(weights)
        model.to(device=device, dtype=dtype)

    runner = ModelRunner(model, cfg)
    bm = BlockManager(config.num_gpu_blocks, config.block_size)
    dlog("memory", f"BlockManager: {config.num_gpu_blocks} blocks x {config.block_size} tokens/block")

    if config.effective_scheduler() == "radix":
        from nano_infer.engine.scheduler.radix_scheduler import RadixScheduler

        sched = RadixScheduler(config)
    else:
        from nano_infer.engine.scheduler.vllm_scheduler import VLLMScheduler

        sched = VLLMScheduler(config)

    return AsyncLLMEngine(runner, sched, bm), runner
