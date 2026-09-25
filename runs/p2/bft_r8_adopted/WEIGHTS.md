# MRFN+（R8 采纳版）权重档案

- **最新采纳模型** = `runs/p2/bft/`（预注册终评 `scripts/eval_p2_test_second.py` 实际加载的路径，
  test 单模型 67.45% / 3-seed 集成 68.64%，commit f7f2286）。
  本目录 `bft_r8_adopted/` 与其 **逐字节相同**（md5 见下，2026-09-25 校验）。
- `checkpoint.pt`：MRFN 头部权重（416,839 参数），已入 git。
- `bert_checkpoint.pt`：完整微调 BERT（109,482,240 参数，438MB），超 GitHub 100MB 单文件限制不入 git。
  其中仅 `bert.encoder.layer.11.*`（16 个 tensor，7,087,872 参数，ft_layers=1）相对
  HF `bert-base-uncased` 有更新，其余 tensor 与公开基线逐位相同（已逐 tensor 校验）。
- `bert_delta_layer11.pt`：仅含被微调的 layer.11。**HF 基线 + delta 可逐位重建完整 bert_checkpoint.pt**
  （3 seeds 全部 round-trip `torch.equal` 验证通过；基线取 ft 键集，忽略基线多余的 cls.* MLM 头）。
- `runs/p2/bft_seed45/`（ft_layers=2）与 `bft_r7_freeze/` 为被拒实验（预注册丢弃，commit 66fade5），
  非采纳模型。

## 校验记录（2026-09-25）

- seed1: full_md5=8a1c33351d4092e7a9b70efb3b6a6197 delta_md5=5363dd2831dd806806cb9c06fd9a9de7 changed=16 tensors, roundtrip_exact=True, delta=27.0MB
- seed2: full_md5=669aa29a241a3da566931fe2fd4b382e delta_md5=6b595477975314372780549fd0e4b4bc changed=16 tensors, roundtrip_exact=True, delta=27.0MB
- seed3: full_md5=faf45f9839a95c97bf459d2607937568 delta_md5=eed7e856e2d63e3d1791275577da808e changed=16 tensors, roundtrip_exact=True, delta=27.0MB
