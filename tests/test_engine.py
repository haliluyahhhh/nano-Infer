import pytest

from nano_infer.config import EngineConfig
from nano_infer.engine.async_llm_engine import build_engine
from nano_infer.engine.sequence import SequenceStatus


@pytest.fixture
def cfg():
    return EngineConfig(
        model_name="dummy",
        max_num_seqs=4,
        max_model_len=512,
        max_num_batched_tokens=256,
        block_size=8,
        num_gpu_blocks=64,
        device="cpu",
        dtype="float32",
        max_tokens=8,
        temperature=0.0,
    )


def test_build_engine(cfg):
    eng, runner = build_engine(cfg)
    assert eng is not None
    assert runner.model is not None


@pytest.mark.asyncio
async def test_generate_single_sequence(cfg):
    eng, _ = build_engine(cfg)
    prompt = [1, 2, 3, 4, 5]
    tokens = []
    async for t in eng.generate_tokens(prompt, max_tokens=5, temperature=0.0):
        tokens.append(t)
    assert len(tokens) >= 1
    assert len(tokens) <= 5


def test_scheduler_prefill_then_decode(cfg):
    eng, _ = build_engine(cfg)
    seq = eng.create_sequence([10, 20, 30], max_tokens=3, temperature=0.0)
    seq.eos_token_id = 999999  # 强制走满 max_tokens
    eng.add_sequence(seq)
    steps = 0
    while seq.status != SequenceStatus.FINISHED and steps < 200:
        eng.step()
        steps += 1
    assert seq.status == SequenceStatus.FINISHED
    assert len(seq.output_ids) == 3
