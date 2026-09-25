"""P3 M3：Integrated Gradients 位置归因层（交付方法；LOO 留作 M4 裁判）。

口径（docs/问题三建模方案.md §4 修订 + M3 预注册 2026-09-25，第二轮审阅修正）：

- **可微输入层级（显式声明）**：text 的 IG 输入 = BERT 输出嵌入层 last_hidden_state
  (50,768)（冻结编码器，eval；集成 = 各 seed 自己的微调 BERT）——绝不对离散 token ID
  做数值插值，α 路径作用在连续嵌入空间 E(α) = E′ + α(E − E′)。audio/vision 的 IG 输入 =
  z-score 标准化后的连续特征 (50,74)/(50,35)。

- **基线语义（第二轮修正，与 v(∅) 严格区分）**：基线 = **全缺失特征值**，且
  **avail 固定为自然观测 o**：text′ = BERT(全 content 位 [MASK]) 嵌入（其 token 序列
  与 v(∅) 的文本输入相同，但 avail 不取 0）、a/v′ = z 空 0 行。解释语义 =
  "给定观测状态，从缺失特征值到实际特征值的贡献"（条件特征归因）。
  为什么不用 avail≡0（= v(∅) 同构）：缺失感知架构（MissingEmbedding 的 where 置零 +
  注意力屏蔽 + 池化排除）使 avail=0 时 ∂F/∂x ≡ 0，实测 IG 全零（2026-09-25 探针
  #01/#13/#18 均为 0.00e+00）——博弈退化为 v≡v(∅) 的平凡博弈、所有位置均为哑玩家。
  因此 **v(∅)（联盟级反事实）只作 Shapley 参照；IG 参照 = 固定 avail 的特征值基线**，
  两层参照系不同属设计选择，不共用名称。

- **三种基线**：missing（上述主基线）/ zero / mean。zero：text=零张量；a/v = raw 0 向量
  过同一标准化 z(0) = −μ/σ。mean：text=训练集 content 位均值嵌入（按 seed BERT）；
  a/v = z 空 0（标准化空间中 train 均值 ⟺ 0，与 missing 数值重合——留痕，敏感性对比由
  text 三基线与 a/v 的 zero 基线承载）。

- **积分（第二轮修正：右端点 → 中点，32 → 64 步）**：中点 Riemann 和
  IG_i = (x_i−x′_i)·(1/n)Σ_{k=1..n} ∂F/∂x_i|_{α=(k−0.5)/n}，n=64。右端点 32 步在
  6 个样本上超 5% 逐样本阈值（O(1/n) 收敛过慢）；中点 32 步已修复 5/6，64 留裕量。
  积分近似误差按**逐样本**检验：|ΣIG − (F(x)−F(x′))| ≤ max(5%·|ΔF|, 1e-3)，
  分类（M2 集成预测类 logit）与回归分别报告。

- **位置→原词聚合（P1 映射 + alpha 断言；数值 = 质量守恒求和）**：P1 `build_grid`
  的 `wp_word_map` 与 `alpha` 实测与附件4 存储网格 20/20 逐 token 对齐（含
  #07/#18 截断样本）。**alpha 的角色**：确定位置→词的归属、片数与截断覆盖，
  并由函数断言词内均匀（α_j = 1/n_w）；**词级数值聚合 = 质量守恒求和**
  A_w = Σ_{j∈w} a_j——α_j 数值不进入 A_w（均匀划分下 A_w 与 α 的具体取值无关，
  Σ_w A_w = Σ_j a_j 保证词级完备性）。special/padding 及 o=0 位永不作证据
  （架构级零归因，逐 seed 验收断言），截断未覆盖的词标记 uncovered。

- **#13 vision 自然缺失**：o_v 全零 → 架构阻断 → IG_v ≡ 0（无证据案例，验收硬检）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.p2.data import CONTRACT
from src.p2.text_mask import mask_text_tokens
from src.p3.shapley import MODS, SampleTensors

N_STEPS = 64
BASELINES = ("missing", "zero", "mean")


def p1_word_grid(raw_text: str, tokenizer=None) -> dict:
    """P1 网格词映射（build_grid 产物，与附件4 存储网格逐 token 对齐，M3 实测 20/20）。

    返回：word_by_position (50,) 每位置所属原词下标（None=special/padding/未映射）、
    alpha (50,) P1 词内均匀权（α_j = 1/n_w，未映射位 0）、n_pieces {w: 保留片数}、
    word_texts、n_words_retained / n_words_all（含被截断整词丢弃的词）。
    """
    from src.p1.grid import build_grid
    g = build_grid(raw_text)
    word_by_pos, n_pieces, word_texts = [None] * CONTRACT.t_grid, {}, {}
    for e in g["wp_word_map"]:
        w = e["word_id"]
        if w is None:
            continue
        word_by_pos[e["position"]] = w
        n_pieces[w] = n_pieces.get(w, 0) + 1
        word_texts[w] = e["word_text"]
    alpha = np.asarray(g["alpha"], dtype=np.float64)
    # 契约断言：词内均匀（P1 alpha 语义）
    for w, cnt in n_pieces.items():
        assert np.isclose(alpha[[j for j, x in enumerate(word_by_pos) if x == w]],
                          1.0 / cnt, atol=1e-9).all(), f"词 {w} alpha 非均匀"
    return {"word_by_position": word_by_pos, "alpha": alpha, "n_pieces": n_pieces,
            "word_texts": word_texts, "n_words_retained": len(n_pieces),
            "n_words_all": len(g["full_word_map"])}


def text_baseline(kind: str, st: SampleTensors, bert_enc, device: str,
                  mean_emb: np.ndarray | None = None) -> torch.Tensor:
    """(1,50,768) 文本基线嵌入。mean 需调用方传入该 seed 的训练均值嵌入 (768,)。

    missing 的 token 序列 = v(∅) 的文本输入，但 IG 路径 avail 固定 o（见模块 docstring）。
    """
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
                avail: dict[str, torch.Tensor], n_steps: int = N_STEPS,
                rule: str = "mid") -> dict:
    """单模型 IG（默认中点 64 步）：各模态 (50,) 位置归因（cls 用 target_cls logit）。

    cudnn 约束：RNN 反传要求 train 标志；MRFN 的 GRU 构造无 dropout，翻转仅满足
    cudnn 后端要求，前向数值不变。
    """
    grads = {m: {o: None for o in ("cls", "reg")} for m in MODS}
    content_t = torch.from_numpy(st.content).to(device)
    rnns = [mod for mod in model.modules() if isinstance(mod, torch.nn.RNNBase)]
    rnn_flags = [r.training for r in rnns]
    for r in rnns:
        r.train()
    for k in range(1, n_steps + 1):
        alpha = k / n_steps if rule == "right" else (k - 0.5) / n_steps
        leaves = {m: (base_feats[m] + alpha * (x_feats[m] - base_feats[m])
                      ).detach().requires_grad_(True) for m in MODS}
        out = model(leaves, content_t, avail)
        for o, target in (("cls", out["logits"][0, target_cls]), ("reg", out["reg"][0])):
            g = torch.autograd.grad(target, list(leaves.values()),
                                    retain_graph=(o == "cls"))
            for m, gm in zip(MODS, g):
                acc = grads[m][o]
                grads[m][o] = gm.detach() if acc is None else acc + gm.detach()
    for r, t in zip(rnns, rnn_flags):
        r.train(t)
    ig = {}
    for m in MODS:
        ig[m] = {o: ((x_feats[m] - base_feats[m])
                     * grads[m][o] / n_steps).sum(dim=-1)[0].cpu().numpy()
                 for o in ("cls", "reg")}
    return ig


def endpoints(model, st, device, target_cls, base_feats, x_feats, avail) -> dict:
    """F(x) 与 F(x′) 之差（cls logit / reg；不参与梯度）。"""
    content_t = torch.from_numpy(st.content).to(device)
    with torch.no_grad():
        fx = model(x_feats, content_t, avail)
        f0 = model(base_feats, content_t, avail)
    return {"cls": (float(fx["logits"][0, target_cls]) - float(f0["logits"][0, target_cls])),
            "reg": (float(fx["reg"][0]) - float(f0["reg"][0]))}


def aggregate_words(pos_attr: np.ndarray, word_grid: dict,
                    content: np.ndarray) -> dict:
    """位置归因 → 原词聚合（P1 alpha 语义，质量守恒）。

    A_w = Σ_{j∈w} a_j（质量守恒求和，Σ_w A_w = Σ_{content} a_j）。
    α（p1_word_grid 提供，词内均匀 1/n_w，函数内已断言）确定归属/片数/覆盖，
    数值不进入 A_w。special/padding/o=0 位不参与（架构级零归因）。
    """
    flat = np.asarray(content).reshape(-1)
    wbp = word_grid["word_by_position"]
    words, leak = {}, float(pos_attr[~flat].sum()) if (~flat).any() else 0.0
    for j, w in enumerate(wbp):
        if not flat[j]:
            continue
        if w is None:               # content 位理应有词；防御留痕
            leak += float(pos_attr[j])
            continue
        words[w] = words.get(w, 0.0) + float(pos_attr[j])
    return {"words": words, "structural_leak": leak, "n_words": len(words),
            "n_pieces": word_grid["n_pieces"], "word_texts": word_grid["word_texts"]}
