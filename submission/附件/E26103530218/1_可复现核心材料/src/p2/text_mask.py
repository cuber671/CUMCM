"""P2 M1-D 文本 [MASK] 重编码与 token id 契约。

契约依据：docs/问题二实施步骤.md §0-拍板2/§1.4 + 建模方案 §3/§4。
- P2 主线文本特征 = 从 text_bert 用自有 frozen bert-base-uncased 现场重编码
  （与附件3 同接口）；附件2 存储的 text(50,768) 不直接进模型（仅作保真对照）。
- 合成缺失：编码**前**把 b_text=1 的 content WordPiece 替换为 [MASK]=103，
  位置/attention/token_type 完全不变；禁止对完整 text 输出做事后置零
  （邻近 token 已吸收被抹除文本的信息，事后置零是错误的缺失语义）。
- 文本缺失位的特征 = [MASK] 的上下文编码（非零，会泄漏邻词上下文——这是
  符号位置缺失的诚实表示），可用性由 a_text=0 表达，模型池化/注意力必须按
  a_text 屏蔽（M1-E 单测：扰动 a=0 位的特征值，模型输出逐位不变）。
- 附件3 text_bert 为 float32：先整数性（|x-round(x)|<1e-3）与值域断言，
  再转 int64，拒绝静默截断；NaN 因比较为 False 同样会被拒。
- BERT 冻结：local_files_only、eval()、参数 requires_grad=False、inference_mode。
本模块依赖 numpy + torch + transformers。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.p2.data import CONTRACT, ROOT, derive_text_masks

MASK_ID = 103          # bert-base-uncased [MASK]
VOCAB_SIZE = 30522
DEFAULT_MODEL_PATH = ROOT / "cache/hf/bert-base-uncased"


def to_int_token_ids(tb: np.ndarray, vocab_size: int = VOCAB_SIZE,
                     tol: float = 1e-3) -> np.ndarray:
    """float/int token id → int64，带整数性与值域断言（契约 §1.4，不得静默截断）。"""
    a = np.asarray(tb)
    if np.issubdtype(a.dtype, np.integer):
        checked = a.astype(np.int64)
    else:
        frac = np.abs(a - np.round(a))
        if not np.all(frac < tol):        # NaN 的比较为 False → 同样在此被拒
            n_bad = int((frac >= tol).sum())
            raise ValueError(f"token id 含非整数值（{n_bad} 位），拒绝静默截断")
        checked = np.round(a).astype(np.int64)
    if checked.size and (int(checked.min()) < 0 or int(checked.max()) >= vocab_size):
        raise ValueError(f"token id 越界 [0,{vocab_size}): "
                         f"[{int(checked.min())},{int(checked.max())}]")
    return checked


def mask_text_tokens(text_bert_int: np.ndarray, b_text: np.ndarray,
                     contract: Contract = CONTRACT, mask_id: int = MASK_ID) -> np.ndarray:
    """编码前替换：b_text=1 的 content WordPiece → [MASK]，其余一切保持不变。

    返回新数组（不修改输入）。b 越界（special/padding 位 b=1）直接拒绝——
    与 §1.2"非 content 位永不合成抹除"一致。
    """
    tb = np.asarray(text_bert_int)
    b = np.asarray(b_text).astype(bool)
    if tb.shape[0] != b.shape[0] or tb.shape[-1] != b.shape[1]:
        raise ValueError(f"text_bert {tb.shape} 与 b_text {b.shape} 前后维不匹配")
    m = derive_text_masks(tb, contract)         # 附带 CLS/SEP/attention/token_type 结构校验
    n_bad = int((b & ~m.content).sum())
    if n_bad:
        raise ValueError(f"b_text 越界：special/padding 位出现 {n_bad} 个抹除位")
    out = tb.copy()
    out[:, 0, :][b] = mask_id
    return out


def load_frozen_bert(model_path: Path = DEFAULT_MODEL_PATH,
                     device: str | None = None) -> torch.nn.Module:
    """加载冻结 BERT（P1 同款约定：本地路径、local_files_only、eval、不训练）。"""
    from transformers import AutoModel
    dev = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(str(model_path), local_files_only=True)
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(dev).eval()


@torch.inference_mode()
def encode_text(text_bert_int: np.ndarray, model: torch.nn.Module,
                contract: Contract = CONTRACT, batch_size: int = 128) -> np.ndarray:
    """(N,3,T) int64 → (N,T,768) float32 last_hidden_state（冻结前向，含 special/pad 行）。"""
    tb = np.asarray(text_bert_int)
    if tb.ndim != 3 or tb.shape[1] != 3 or not np.issubdtype(tb.dtype, np.integer):
        raise ValueError(f"期望 (N,3,T) 整型 text_bert，实际 {tb.shape} {tb.dtype}")
    if tb.shape[2] != contract.t_grid:
        raise ValueError(f"T={tb.shape[2]} 与契约 {contract.t_grid} 不符")
    device = next(model.parameters()).device
    outs = []
    for i in range(0, tb.shape[0], batch_size):
        chunk = tb[i:i + batch_size]
        batch = {name: torch.from_numpy(chunk[:, j, :].astype(np.int64)).to(device)
                 for j, name in enumerate(("input_ids", "attention_mask", "token_type_ids"))}
        hidden = model(**batch).last_hidden_state
        outs.append(hidden.detach().cpu().numpy().astype(np.float32))
    res = np.concatenate(outs, axis=0) if outs else np.zeros((0, contract.t_grid, 768), np.float32)
    if not np.isfinite(res).all():
        raise RuntimeError("BERT 输出含 NaN/Inf")
    return res


class BertTextEncoder(torch.nn.Module):
    """可解冻的 BERT 文本编码器（Round 7：最后 N 层参与训练，其余冻结）。

    - 默认 unfreeze_last=0 等价于冻结前向；
    - 解冻层保持低学习率（调用方配置参数组，建议 5e-5）；
    - BERT 内部保持 eval（dropout 关闭）：小样本下关闭随机失活更稳；
    - forward 直接接收 (N,3,T) 整型 text_bert（支持 autograd，供微调反向传播）。
    """

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH, unfreeze_last: int = 0):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(str(model_path), local_files_only=True)
        for p in self.bert.parameters():
            p.requires_grad_(False)
        n = max(0, int(unfreeze_last))
        n_layers = len(self.bert.encoder.layer)
        if n > n_layers:
            raise ValueError(f"unfreeze_last={n} 超过总层数 {n_layers}")
        for layer in self.bert.encoder.layer[-n:] if n else []:
            for p in layer.parameters():
                p.requires_grad_(True)
        self.unfreeze_last = n
        self.bert.eval()

    def unfrozen_parameters(self):
        return [p for p in self.bert.parameters() if p.requires_grad]

    def forward(self, tb_int: torch.Tensor) -> torch.Tensor:
        batch = {"input_ids": tb_int[:, 0], "attention_mask": tb_int[:, 1],
                 "token_type_ids": tb_int[:, 2]}
        return self.bert(**batch).last_hidden_state
