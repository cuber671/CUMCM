"""P3 M3：Integrated Gradients 位置归因层（交付方法；LOO 留作 M4 裁判）。

口径（docs/问题三建模方案.md §4 + M3 预注册 2026-09-25）：
- **可微输入层级（显式声明）**：text 的 IG 输入 = BERT 输出嵌入层 last_hidden_state
  (50,768)（冻结编码器，eval；集成 = 各 seed 自己的微调 BERT）——绝不对离散 token ID
  做数值插值，α 路径作用在连续嵌入空间 E(α) = E′ + α(E − E′)。audio/vision 的 IG 输入 =
  z-score 标准化后的连续特征 (50,74)/(50,35)；
- **全缺失基线（主基线，与 v(∅) 同构）**：text = BERT(全 content 位 [MASK]) 嵌入（即
  v(∅) 的文本输入本身）；audio/vision = z 空 0 行（raw μ；v(∅) 内部置零路径同构）；
  avail 固定为自然观测 o（模型已冻结，δ 阈值硬切，不做 δ 插值）；
- **三基线敏感性**：missing / zero / mean。zero：text=零张量；a/v = raw 0 向量过同一
  标准化 z(0) = −μ/σ。mean：text=训练集 content 位均值嵌入（按 seed BERT）；a/v = z 空 0
  （标准化空间中 train 均值 ⟺ 0，与 missing 数值重合——留痕，敏感性对比由 text 三基线
  与 a/v 的 zero 基线承载）；
- **32 步积分**：右端点 Riemann 和 IG_i = (x_i−x′_i)·(1/32)Σ_{k=1..32} ∂F/∂x_i|_{α=k/32}；
  积分近似误差 = |ΣIG − (F(x)−F(x′))|，分类（预测类 logit，类索引取 M2 集成预测）与
  回归分别报告；
- **位置→原词聚合**：先算 50 模型位置的 a_{m,j}（维内求和），再按 P1 alpha 语义聚合
  （同一原词的 WordPiece 均匀权，word_attr = Σ_{pieces} a_j；special/padding 永不作证据，
  其归因单列 structural_leak；截断未覆盖的词标记 uncovered）；
- **#13 vision 自然缺失**：o_v 全零 → MissingEmbedding/注意力/池化全部阻断 →
  ∂F/∂v ≡ 0 → IG_v ≡ 0（无证据案例，验收硬检）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.p2.data import CONTRACT
from src.p2.text_mask import mask_text_tokens
from src.p3.shapley import MODS, SampleTensors

N_STEPS = 32
BASELINES = ("missing", "zero", "mean")


def word_piece_map(raw_text: str, tokenizer) -> list:
    """(50,) 每位置所属原词下标；special/padding = None。分词规则与 M1 契约一致。"""
    enc = tokenizer(raw_text, truncation=True, max_length=50,
                    return_offsets_mapping=True)
    wids = enc.word_ids()              # len ≤ 50，None=特殊位
    assert len(wids) <= CONTRACT.t_grid
    return list(wids) + [None] * (CONTRACT.t_grid - len(wids))


def text_baseline(kind: str, st: SampleTensors, bert_enc, device: str,
                  mean_emb: np.ndarray | None = None) -> torch.Tensor:
    """(1,50,768) 文本基线嵌入。mean 需调用方传入该 seed 的训练均值嵌入 (768,)。"""
    if kind == "missing":
        ids = torch.from_numpy(
            mask_text_tokens(st.text_bert, st.content, CONTRACT)).to(device)
        with torch.no_grad():
            return bert_enc(ids)
    if kind == "zero":
        return torch.zeros((1, CONTRACT.t_grid, 768), device=device)
    if kind == "mean":
        assert mean_emb is not None, "mean 基线需传入训练 content 均值嵌入"
        return torch.from_numpy(np.tile(mean_emb, (1, CONTRACT.t_grid, 1)).astype(
            np.float32)).to(device)
    raise ValueError(kind)


def av_baseline(kind: str, mod: str, stats: dict) -> np.ndarray:
    """(50,D) 连续模态基线（z 空间）。missing/mean=0 行（后者与前者数值重合，留痕）。"""
    mu = np.asarray(stats[mod]["mean"], dtype=np.float64)
    sd = np.asarray(stats[mod]["std"], dtype=np.float64)
    if kind == "zero":                       # raw 0 向量过同一标准化
        row = (0.0 - mu) / np.where(sd < 1e-8, 1.0, sd)
    elif kind in ("missing", "mean"):
        row = np.zeros_like(mu)
    else:
        raise ValueError(kind)
    return np.tile(row[None], (CONTRACT.t_grid, 1)).astype(np.float32)


def train_mean_text_embedding(bert_enc, text_bert_train: np.ndarray,
                              content_train: np.ndarray, device: str,
                              batch: int = 128) -> np.ndarray:
    """训练集全部 content 位嵌入均值 (768,)（mean 基线；按 seed 各算各的）。"""
    sums, cnt = torch.zeros(768, device=device, dtype=torch.float64), 0
    with torch.no_grad():
        for i in range(0, text_bert_train.shape[0], batch):
            ids = torch.from_numpy(text_bert_train[i:i + batch]).to(device)
            h = bert_enc(ids).to(torch.float64)                 # (B,50,768)
            c = torch.from_numpy(content_train[i:i + batch]).to(device)
            sums += (h * c[..., None]).sum(dim=(0, 1))
            cnt += int(c.sum())
    return (sums / max(cnt, 1)).cpu().numpy().astype(np.float32)


def ig_modality(model, st: SampleTensors, device: str, target_cls: int,
                base_feats: dict[str, torch.Tensor], x_feats: dict[str, torch.Tensor],
                avail: dict[str, torch.Tensor], n_steps: int = N_STEPS) -> dict:
    """单模型 32 步 IG：返回各模态 (50,) 位置归因 + ΔF（cls logit / reg）。"""
    grads = {m: {o: None for o in ("cls", "reg")} for m in MODS}
    content_t = torch.from_numpy(st.content).to(device)
    # cudnn 约束：RNN 反传要求 train 标志；MRFN 的 GRU 无 dropout（构造无该参），
    # 翻转仅满足 cudnn 后端要求，前向数值不变（eval 其余模块保持无 dropout）
    rnns = [mod for mod in model.modules() if isinstance(mod, torch.nn.RNNBase)]
    rnn_flags = [r.training for r in rnns]
    for r in rnns:
        r.train()
    for k in range(1, n_steps + 1):
        alpha = k / n_steps
        leaves = {m: (base_feats[m] + alpha * (x_feats[m] - base_feats[m])
                      ).detach().requires_grad_(True) for m in MODS}
        out = model(leaves, content_t, avail)
        for o, target in (("cls", out["logits"][0, target_cls]), ("reg", out["reg"][0])):
            g = torch.autograd.grad(target, list(leaves.values()),
                                    retain_graph=(o == "cls"))
            for m, gm in zip(MODS, g):
                acc = grads[m][o]
                grads[m][o] = gm.detach() if acc is None else acc + gm.detach()
        # ΔF 参照由 endpoints() 单独计算（端点不进梯度路径）
    for r, t in zip(rnns, rnn_flags):
        r.train(t)
    ig = {}
    for m in MODS:
        ig[m] = {o: ((x_feats[m] - base_feats[m])
                     * grads[m][o] / n_steps).sum(dim=-1)[0].cpu().numpy()
                 for o in ("cls", "reg")}
    return ig


def endpoints(model, st, device, target_cls, base_feats, x_feats, avail) -> dict:
    """F(x) 与 F(x′)（cls logit / reg；不参与梯度）。"""
    content_t = torch.from_numpy(st.content).to(device)
    with torch.no_grad():
        fx = model(x_feats, content_t, avail)
        f0 = model(base_feats, content_t, avail)
    return {"cls": (float(fx["logits"][0, target_cls]) - float(f0["logits"][0, target_cls])),
            "reg": (float(fx["reg"][0]) - float(f0["reg"][0]))}


def aggregate_words(pos_attr: np.ndarray, wp_words: list,
                    content: np.ndarray) -> dict:
    """位置归因 → 原词聚合（P1 alpha 语义：词内均匀，word = Σ pieces）。

    返回 {word_idx: attr}、special/pad 合计（structural_leak）、uncovered 词数。
    """
    flat = np.asarray(content).reshape(-1)
    words, leak = {}, float(pos_attr[~flat].sum()) if (~flat).any() else 0.0
    for j, w in enumerate(wp_words):
        if not flat[j]:
            continue
        if w is None:               # content 位理应都有词；防御留痕
            leak += float(pos_attr[j])
            continue
        words[w] = words.get(w, 0.0) + float(pos_attr[j])
    return {"words": words, "structural_leak": leak, "n_words": len(words)}
