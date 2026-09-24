"""P2 M1-D 单测：token id 断言、[MASK] 替换、冻结 BERT 前向。

M1-E 的"扰动 a=0 位特征、模型输出逐位不变"属模型级测试，不在本文件；
本文件验证输入侧契约：b 只落 content、特殊位不变、dtype 断言、前向确定性。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.p2.data import Contract
from src.p2.text_mask import (DEFAULT_MODEL_PATH, MASK_ID, VOCAB_SIZE,
                              encode_text, load_frozen_bert, mask_text_tokens,
                              to_int_token_ids)

T = 16
SMALL = Contract(counts={"train": 1, "valid": 1, "test": 1}, d_audio=3, d_vision=2,
                 d_text=5, t_grid=T, labels=(0, 1, 2), rl_range=(-3.0, 3.0),
                 cls_counts_train={0: 1, 1: 0, 2: 0})


def make_tb(n=3, seed=0):
    """(n,3,T) 合法 text_bert：CLS + 2..k 个 wp + SEP + pad。"""
    rng = np.random.default_rng(seed)
    tb = np.zeros((n, 3, T), dtype=np.int64)
    for i in range(n):
        k = int(rng.integers(2, T - 3))          # content 数 ≥2
        sep = k + 1
        tb[i, 0, 0], tb[i, 0, sep] = 101, 102
        tb[i, 0, 1:k + 1] = rng.integers(1000, 5000, size=k)
        tb[i, 1, :sep + 1] = 1
    return tb


def make_b(tb, seed=1, dense=False):
    """content 上的合成 b（dense=True 时全部 content 位=1）。"""
    from src.p2.data import derive_text_masks
    m = derive_text_masks(tb, SMALL)
    rng = np.random.default_rng(seed)
    b = m.content & (rng.random(m.content.shape) < 0.5 if not dense else True)
    return b


# ---- token id 断言 ----

def test_to_int_passthrough_and_cast():
    tb = make_tb(2)
    assert to_int_token_ids(tb).dtype == np.int64
    f32 = tb.astype(np.float32)                   # 精确可表的 float → 通过
    assert (to_int_token_ids(f32) == tb).all()
    near = tb.astype(np.float64) + 1e-6           # 近整 → 通过
    assert (to_int_token_ids(near) == tb).all()


@pytest.mark.parametrize("bad", ["frac", "neg", "oob", "nan"])
def test_to_int_rejects(bad):
    tb = make_tb(1).astype(np.float32)
    if bad == "frac":
        tb[0, 0, 2] = 1234.5                      # .5 远离整数
    elif bad == "neg":
        tb[0, 0, 2] = -3.0
    elif bad == "oob":
        tb[0, 0, 2] = VOCAB_SIZE + 5
    elif bad == "nan":
        tb[0, 0, 2] = np.nan
    with pytest.raises(ValueError):
        to_int_token_ids(tb)


# ---- [MASK] 替换 ----

def test_mask_only_content_positions():
    tb = make_tb(3)
    b = make_b(tb, dense=True)
    out = mask_text_tokens(tb, b, contract=SMALL)
    from src.p2.data import derive_text_masks
    m = derive_text_masks(tb, SMALL)
    assert (out[:, 0, :][b] == MASK_ID).all()     # b 位全变 [MASK]
    assert (out[:, 0, :][~b] == tb[:, 0, :][~b]).all()
    assert (out[:, 1, :] == tb[:, 1, :]).all()    # attention 不变
    assert (out[:, 2, :] == tb[:, 2, :]).all()    # token_type 不变
    assert (out[:, 0, 0] == 101).all()            # CLS 不变
    sep = m.sep_pos
    assert (out[np.arange(3), 0, sep] == 102).all()  # SEP 不变
    assert (out[:, 0, :][~m.content] == tb[:, 0, :][~m.content]).all()  # pad 不变
    assert tb is not out                                        # 返回新数组
    assert (tb[:, 0, :][b] != MASK_ID).all()      # 输入未被修改（ids 抽自 1000-5000）


def test_mask_rejects_out_of_content():
    tb = make_tb(2)
    b = np.zeros((2, T), dtype=bool)
    b[0, 0] = True                                # CLS 位
    with pytest.raises(ValueError):
        mask_text_tokens(tb, b, contract=SMALL)
    b2 = np.zeros((2, T), dtype=bool)
    b2[1, -1] = True                              # padding 位
    with pytest.raises(ValueError):
        mask_text_tokens(tb, b2, contract=SMALL)
    b3 = np.zeros((3, T), dtype=bool)             # 形状不匹配
    with pytest.raises(ValueError):
        mask_text_tokens(tb, b3, contract=SMALL)


def test_mask_noop_when_no_b():
    tb = make_tb(2)
    out = mask_text_tokens(tb, np.zeros((2, T), dtype=bool), contract=SMALL)
    assert (out == tb).all()


# ---- 冻结 BERT 前向（本地模型缺失则跳过）----

@pytest.fixture(scope="module")
def bert():
    if not (DEFAULT_MODEL_PATH / "config.json").exists():
        pytest.skip("本地 BERT 不存在")
    return load_frozen_bert(device="cpu")


def test_frozen_bert_properties(bert):
    assert not bert.training
    assert not any(p.requires_grad for p in bert.parameters())


def test_encode_shapes_and_determinism(bert):
    tb = make_tb(4, seed=3)
    h1 = encode_text(tb, bert, contract=SMALL, batch_size=2)
    h2 = encode_text(tb, bert, contract=SMALL, batch_size=2)
    h3 = encode_text(tb, bert, contract=SMALL, batch_size=3)   # 不同分批
    assert h1.shape == (4, T, 768) and h1.dtype == np.float32
    assert np.isfinite(h1).all()
    assert (h1 == h2).all()                       # 同分批逐位确定
    assert np.allclose(h1, h3, atol=1e-5)         # 跨分批仅浮点归约序差


def test_masked_encode_changes_output(bert):
    """[MASK] 重编码：b=1 位的输出必须改变（事后置零是错误语义的前车之鉴）。"""
    tb = make_tb(2, seed=5)
    b = make_b(tb, seed=6, dense=False)
    assert b.any()
    clean = encode_text(tb, bert, contract=SMALL)
    masked = encode_text(mask_text_tokens(tb, b, contract=SMALL), bert, contract=SMALL)
    assert np.abs(masked[b] - clean[b]).max() > 1e-6
