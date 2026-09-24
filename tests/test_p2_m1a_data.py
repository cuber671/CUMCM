"""P2 M1-A 单测：数据契约（synthetic 构造为主 + 真实附件2 集成一条）。

覆盖 M1-A 审阅要点：
- 掩码分区互斥完备、SEP/满长边界
- 严格零语义（1e-30 非零仍是观测；0 与 -0.0 是缺失）
- o_text ≡ 1 不受 text 数值影响
- 统计量 train-only、只含 o=1 位置、常量维 std→1.0
- validate 对各类污染逐一报警
- build 产物完整性
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.p2.data import (CONTRACT, STD_EPS, Contract, ALIGNED_PKL, build_m1a,
                         compute_stats, derive_observed, derive_text_masks,
                         validate_aligned)

TC, DA, DV, DT = 8, 3, 2, 5


def make_split(rng, n, prefix="s"):
    """构造合法 split：text_bert 结构合法，audio/vision 含指定严格零行，
    非 content 位（special+padding）与附件2 一致地严格全零。"""
    iid = np.zeros((n, 3, TC), dtype=np.int64)
    zero_a = np.zeros((n, TC), dtype=bool)   # 待置零的 content 行
    zero_v = np.zeros((n, TC), dtype=bool)
    sep_pos = np.full(n, TC - 1)
    for i in range(n):
        k = int(rng.integers(1, TC - 1))     # content 数
        sp = k + 1
        sep_pos[i] = sp
        iid[i, 0, 0] = 101
        iid[i, 0, 1:k + 1] = rng.integers(1000, 5000, size=k)
        iid[i, 0, sp] = 102
        iid[i, 1, :sp + 1] = 1
    pos = np.arange(TC)[None, :]
    content = (iid[:, 1, :] == 1) & (pos != 0) & (pos != sep_pos[:, None])
    cidx = [np.where(content[i])[0] for i in range(n)]
    for i in range(n):
        if len(cidx[i]) and rng.random() < 0.5:
            zero_a[i, cidx[i][0]] = True
        if len(cidx[i]) and rng.random() < 0.5:
            zero_v[i, cidx[i][0]] = True
    audio = rng.normal(size=(n, TC, DA))
    vision = rng.normal(size=(n, TC, DV))
    audio[zero_a] = 0.0
    vision[zero_v] = 0.0
    audio[~content] = 0.0   # 附件2 既定结构：special/padding 位严格全零
    vision[~content] = 0.0
    text = rng.normal(size=(n, TC, DT)).astype(np.float32)
    cls = rng.integers(0, 3, size=n).astype(np.float64)
    rl = rng.uniform(-3, 3, size=n)
    return {
        "id": [f"{prefix}sid{i}" for i in range(n)],
        "raw_text": ["hello world"] * n,
        "text_bert": iid, "text": text, "audio": audio.astype(np.float64),
        "vision": vision.astype(np.float64),
        "classification_labels": cls, "regression_labels": rl,
    }


@pytest.fixture(scope="module")
def att_small():
    rng = np.random.default_rng(7)
    a = {"train": make_split(rng, 6, "t"), "valid": make_split(rng, 2, "v"),
         "test": make_split(rng, 2, "e")}
    counts = {int(k): int(v) for k, v in
              zip(*np.unique(a["train"]["classification_labels"], return_counts=True))}
    small = Contract(counts={"train": 6, "valid": 2, "test": 2}, d_audio=DA, d_vision=DV,
                     d_text=DT, t_grid=TC, labels=(0, 1, 2), rl_range=(-3.0, 3.0),
                     cls_counts_train=counts)
    return a, small


# ---- 掩码分区 ----

def test_masks_partition(att_small):
    a, c = att_small
    m = derive_text_masks(a["train"]["text_bert"], c)
    assert not (m.content & m.special).any() and not (m.content & m.padding).any()
    assert not (m.special & m.padding).any()
    assert (m.content | m.special | m.padding).all()
    iid = a["train"]["text_bert"][:, 0, :]
    assert (iid[np.arange(6), m.sep_pos] == 102).all()
    # special 恰为 CLS+SEP（每样本 2 个）
    assert (m.special.sum(axis=1) == 2).all()


def test_masks_full_length(att_small):
    """无 padding 的满长样本：SEP@T-1，padding 全 False。"""
    a, c = att_small
    iid = a["train"]["text_bert"].copy()
    full = np.zeros((1, 3, TC), dtype=np.int64)
    full[0, 0, 0], full[0, 0, -1] = 101, 102
    full[0, 0, 1:-1] = 4000
    full[0, 1, :] = 1
    m = derive_text_masks(full, c)
    assert m.sep_pos[0] == TC - 1 and not m.padding.any()
    assert m.content.sum() == TC - 2


def test_masks_rejects_bad_structure(att_small):
    _, c = att_small
    bad = np.zeros((1, 3, TC), dtype=np.int64)
    bad[0, 0, 0] = 999  # 非 CLS
    with pytest.raises(ValueError):
        derive_text_masks(bad, c)
    bad2 = np.zeros((1, 3, TC), dtype=np.int64)
    bad2[0, 0, 0], bad2[0, 0, 3] = 101, 500  # SEP 位不是 [SEP] id
    bad2[0, 1, :4] = 1
    with pytest.raises(ValueError):
        derive_text_masks(bad2, c)
    bad3 = np.zeros((1, 3, TC), dtype=np.int64)
    bad3[0, 0, 0], bad3[0, 0, 3] = 101, 102
    bad3[0, 1, :4] = 1
    bad3[0, 1, 2] = 2  # attention_mask 非二值
    with pytest.raises(ValueError):
        derive_text_masks(bad3, c)


# ---- 自然观测 ----

def test_strict_zero_semantics(att_small):
    a, c = att_small
    m = derive_text_masks(a["train"]["text_bert"], c)
    o = derive_observed(a["train"]["audio"], a["train"]["vision"], m.content)
    # 构造：content 行 [1e-30] 仍观测；[0.0] 与 [-0.0] 缺失
    aud = a["train"]["audio"].copy()
    cm = m.content
    p = np.where(cm)
    aud[p[0][0], p[1][0], :] = 1e-30
    aud[p[0][1], p[1][1], :] = 0.0
    aud[p[0][2], p[1][2], :] = -0.0
    o2 = derive_observed(aud, a["train"]["vision"], m.content)
    assert o2["o_audio"][p[0][0], p[1][0]] == 1
    assert o2["o_audio"][p[0][1], p[1][1]] == 0
    assert o2["o_audio"][p[0][2], p[1][2]] == 0
    # 非 content 位一律 0（2026-09-24 拍板，与 P1 observed_mask 语义统一）
    assert not o2["o_audio"][~cm].any()
    assert np.array_equal(o["o_text"], cm) and np.array_equal(o2["o_text"], cm)


def test_o_zero_outside_content(att_small):
    """非 content 位的 o_m 三模态一律 0（M1-C 依赖该性质保证特殊位永不抹除）。"""
    a, c = att_small
    m = derive_text_masks(a["train"]["text_bert"], c)
    o = derive_observed(a["train"]["audio"], a["train"]["vision"], m.content)
    outside = ~m.content
    for k in ("o_text", "o_audio", "o_vision"):
        assert not o[k][outside].any(), k


def test_o_text_scoped_to_content(att_small):
    """o_text = content_mask：附件2 文本 content 位无真实缺失；
    padding 行是 [PAD] 的 BERT 编码（100% 非零），"行非零"不构成观测证据。"""
    a, c = att_small
    m = derive_text_masks(a["train"]["text_bert"], c)
    o = derive_observed(a["train"]["audio"], a["train"]["vision"], m.content)
    assert np.array_equal(o["o_text"], m.content)


# ---- 统计量 ----

def test_stats_train_only_and_scope(att_small):
    a, c = att_small
    m = derive_text_masks(a["train"]["text_bert"], c)
    o = derive_observed(a["train"]["audio"], a["train"]["vision"], m.content)
    inc = m.content & o["o_audio"]
    s1 = compute_stats(a["train"]["audio"], inc)
    b = copy.deepcopy(a)                                   # 不污染共享 fixture
    b["valid"]["audio"] = b["valid"]["audio"] * 1000 + 7   # 扰动 valid 不影响 train 统计
    m2 = derive_text_masks(b["train"]["text_bert"], c)
    o2 = derive_observed(b["train"]["audio"], b["train"]["vision"], m2.content)
    s2 = compute_stats(b["train"]["audio"], m2.content & o2["o_audio"])
    assert np.array_equal(s1.mean, s2.mean)
    # 统计只覆盖 include 位置
    vals = a["train"]["audio"][inc]
    assert s1.n_positions == vals.shape[0]
    assert np.allclose(s1.mean, vals.astype(np.float64).mean(0))
    assert np.allclose(s1.std_raw, vals.astype(np.float64).std(0))


def test_stats_constant_dim_eps_rule():
    x = np.zeros((4, 5, 4))
    x[:, :, 0] = 2.5                                   # 常量维 std=0 → 裁
    x[:, :, 1] = 1e-12                                 # 全零近似常量 std=0 → 裁
    x[:, :, 2] = np.linspace(-1e-9, 1e-9, 5)           # std<1e-8 → 裁
    x[:, :, 3] = np.linspace(-1.0, 1.0, 5)             # 正常维 → 保留
    s = compute_stats(x, np.ones((4, 5), dtype=bool))
    assert s.clipped_dims == [0, 1, 2]
    assert (s.std[[0, 1, 2]] == 1.0).all()
    assert 3 not in s.clipped_dims and s.std[3] > 0
    assert s.mean[0] == pytest.approx(2.5)


def test_stats_eps_threshold_exact():
    x = np.zeros((10, 1, 2))
    x[:, 0, 0] = np.linspace(-5e-9, 5e-9, 10)          # std < 1e-8 → 裁
    x[:, 0, 1] = np.linspace(-5e-7, 5e-7, 10)          # std > 1e-8 → 保留
    s = compute_stats(x, np.ones((10, 1), dtype=bool))
    assert s.clipped_dims == [0]
    assert s.std[1] == pytest.approx(s.std_raw[1])


# ---- validate 报警 ----

def _corrupt(a, kind):
    b = copy.deepcopy(a)
    if kind == "nan":
        b["train"]["audio"][0, 0, 0] = np.nan
    elif kind == "dup_id":
        b["valid"]["id"][1] = b["valid"]["id"][0]
    elif kind == "cross_id":
        b["test"]["id"][0] = b["train"]["id"][0]
    elif kind == "bad_label":
        b["train"]["classification_labels"][0] = 3.0
    elif kind == "bad_rl":
        b["test"]["regression_labels"][0] = 5.0
    elif kind == "noncontent_nonzero":
        b["train"]["audio"][0, -1, :] = 1.0            # padding 位非零
    elif kind == "wrong_count":
        b["train"]["id"] = b["train"]["id"][:-1]
        b["train"]["audio"] = b["train"]["audio"][:-1]
    elif kind == "float_text_bert":
        b["train"]["text_bert"] = b["train"]["text_bert"].astype(np.float32)
    elif kind == "short_raw_text":
        b["train"]["raw_text"] = b["train"]["raw_text"][:-1]
    elif kind == "short_rl":
        b["test"]["regression_labels"] = b["test"]["regression_labels"][:-1]
    elif kind == "nan_rl":
        b["train"]["regression_labels"][0] = np.nan
    elif kind == "attention2":
        b["train"]["text_bert"][0, 1, 2] = 2
    elif kind == "extra_split":
        b["holdout"] = dict(b["test"])
    elif kind == "missing_split":
        del b["valid"]
    elif kind == "extra_field":
        b["train"]["junk"] = 1
    return b


@pytest.mark.parametrize("kind", ["nan", "dup_id", "cross_id", "bad_label", "bad_rl",
                                  "noncontent_nonzero", "wrong_count", "float_text_bert",
                                  "short_raw_text", "short_rl", "nan_rl", "attention2",
                                  "extra_split", "missing_split", "extra_field"])
def test_validate_catches(att_small, kind):
    a, c = att_small
    base = validate_aligned(a, c)
    assert all(x.ok for x in base), [x.name for x in base if not x.ok]
    bad = validate_aligned(_corrupt(a, kind), c)
    failed = [x.name for x in bad if not x.ok]
    assert failed, f"{kind} 未被任何检查发现"
    if kind == "extra_field":
        assert "fields_exact" in failed


# ---- build 产物 ----

def test_build_m1a_artifacts(att_small, tmp_path):
    import hashlib
    a, c = att_small
    src = tmp_path / "fake_source.pkl"
    src.write_bytes(b"m1a-source")
    rep = build_m1a(a, tmp_path, contract=c, source_path=src)
    assert rep["all_checks_pass"]
    assert rep["artifacts_written"] is True
    # 来源指纹必须来自显式传入的 source_path，而非模块常量
    assert rep["source"]["path"] == str(src)
    assert rep["source"]["sha256"] == hashlib.sha256(b"m1a-source").hexdigest()
    assert (tmp_path / "normalization_stats.npz").exists()
    assert (tmp_path / "m1a_state.npz").exists()
    assert (tmp_path / "m1a_report.json").exists()
    st0 = np.load(tmp_path / "m1a_state.npz")
    assert str(st0["source_sha256"]) == rep["source"]["sha256"]
    z = np.load(tmp_path / "normalization_stats.npz")
    assert z["mean_audio"].shape == (DA,) and z["std_vision"].shape == (DV,)
    st = np.load(tmp_path / "m1a_state.npz")
    for sp in ("train", "valid", "test"):
        assert st[f"{sp}_content"].shape == (c.counts[sp], TC)
        assert set(np.unique(st[f"{sp}_o_audio"])) <= {0, 1}
    # b≡0/a≡o 基础态由 notes 声明；state 中 o 即最终 a 的底座
    assert rep["o_summary"]["train"]["vision"]["n_o0"] >= 0


# ---- 失败不落产物 ----

def test_build_failure_writes_no_artifacts(att_small, tmp_path):
    """校验失败：只写失败报告（artifacts_written=false），不留下可消费的 npz。"""
    a, c = att_small
    b = copy.deepcopy(a)
    b["train"]["audio"][0, 0, 0] = np.nan
    rep = build_m1a(b, tmp_path, contract=c)
    assert not rep["all_checks_pass"]
    assert rep["artifacts_written"] is False
    assert (tmp_path / "m1a_report.json").exists()
    assert not (tmp_path / "normalization_stats.npz").exists()
    assert not (tmp_path / "m1a_state.npz").exists()
    rep_json = json.loads((tmp_path / "m1a_report.json").read_text(encoding="utf-8"))
    assert rep_json["artifacts_written"] is False


# ---- 真实附件2 集成（文件缺失则跳过）----

def test_real_attachment2_validate():
    if not ALIGNED_PKL.exists():
        pytest.skip("附件2 不存在")
    from src.p2.data import load_aligned
    att = load_aligned()
    checks = validate_aligned(att, CONTRACT)
    failed = [(c.name, c.detail) for c in checks if not c.ok]
    assert not failed, f"真实附件2 校验失败: {failed}"
