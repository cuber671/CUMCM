# 可复现核心材料

比赛："华为杯"第二十三届中国研究生数学建模竞赛  
赛题：E 题 — 复杂场景下多模态情感预测的数学建模与算法设计

---

## 一、内容组织

```
1_可复现核心材料/
├── README.md                      ← 本文件（运行说明 + 数据集处理规则 + 参数配置）
├── src/                           ← 完整源代码
│   ├── p1/                        ← 问题一：CTC 强制对齐、50 位网格坐标、自提取音视特征
│   ├── p2/                        ← 问题二：MRFN 模型、合成缺失算子、31 场景评测、消融
│   └── p3/                        ← 问题三：8 联盟精确 Shapley、64 步中点积分梯度、保真度检验
├── scripts/                       ← 28 个入口脚本（按问题一二三分组）
├── models/                        ← MRFN+ 三种子模型权重（checkpoint.pt × 3）
├── data/
│   └── p1_features/               ← 问题一自生成的 100 条样本多模态时序特征
│       ├── features.pkl           ← 100 条样本汇总特征矩阵（npz 形式）
│       ├── samples/               ← 100 条单样本特征（每条一个 .pkl）
│       └── manifest.csv           ← 样本清单与契约字段
├── requirements.in                ← 直接依赖清单
├── requirements.lock              ← 全量依赖版本锁定
└── environment-manifest.json      ← 运行环境契约（Python/各工具版本指纹）
```

## 二、运行环境

| 组件 | 版本 |
|---|---|
| Python | 3.10 |
| PyTorch / torchaudio | 2.11.0 / 2.11.0 |
| transformers | 5.17.0 |
| openSMILE | 2.6.0（eGeMAPSv02） |
| Py-Feat | 0.6.1 |
| imageio-ffmpeg / ffmpeg | 0.6.0 / 7.0.2（static） |

依赖安装：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
```

`bert-base-uncased`（revision 86b5e09）由 transformers 自动下载至 `cache/`；其余权重随脚本首次运行补齐。

## 三、数据集处理规则

### 3.1 原始素材（赛题提供）

- **附件1**：100 条原始英文视频（37 个 video_id 子文件夹，时长 2.648–34.567 s）及标注表
- **附件2**：4850 条标准化多模态特征文件（文本 768 / 语音 74 / 视觉 35 维），按训练/验证/测试 = 3395/728/727 组织；附件3 与附件4 无标签
- **附件3**：30 条无标签样本的特征文件，含随机模态局部缺失
- **附件4**：20 条无标签真实场景视频及对应三模态特征文件

### 3.2 自生成资产（提交包内）

- `data/p1_features/`：问题一自生成的 100 条样本多模态时序特征，是问题二、问题三模型对齐的"特征对照基准"
  - `features.pkl`：100 条样本的特征矩阵汇总
  - `samples/<video_id>__<idx>.pkl`：单样本特征，便于逐条复核
  - `manifest.csv`：样本编号、字段 schema、契约约束（p1.v2）

### 3.3 数据契约（p1.v2）

输出特征与附件1 标注表一一对应：样本编号无缺失、无重复、无多余，每条样本通过 p1.v2 契约的 schema 校验（形状、数据类型、结构位分区、SEP 位置、观测掩码取值域），违例计数为零。

## 四、参数配置

### 4.1 评测掩码库

| 参数 | 值 | 说明 |
|---|---|---|
| `MASK_LIBRARY_SEED` | 2026 | 评测掩码库种子，与训练种子分离 |
| `K_INSTANCES` | 8 | 每场景 8 个种子化实例 |
| `MIN_AVAIL_RATE` | 0.20 | 任一模态可用率下限，规避三模态同时全缺失 |
| `JOINT_RATE` | 0.40 | 联合缺失场景固定 40% 缺失率 |

### 4.2 模型训练

| 参数 | 值 | 章节 |
|---|---|---|
| 优化器 | AdamW | §5.3 |
| 学习率 | 10⁻³ | §5.3 |
| 权重衰减 | 10⁻⁴ | §5.3 |
| 批大小 | 32 | §5.3 |
| 最大轮数 | 40 | §5.3 |
| 早停耐心 | 8 轮 | §5.3 |
| λ_r（分类/回归权重） | 1 | §5.2 |
| BERT 微调学习率 | 5×10⁻⁵ | §5.5 |
| 集成种子数 | 3（seed1/2/3） | §5.5、§6 |

### 4.3 模型权重

`models/` 目录包含 MRFN+ 三种子的检查点（`MRFN_seed{1,2,3}/checkpoint.pt`），均为附件3、附件4 推理与终评使用的同一权重族；BERT 主干权重由 transformers 自动下载至 `cache/`。

## 五、复现最小链路

按论文附录 A.4 列出的入口脚本顺序运行即可重出全部正文结果：

```bash
# 问题一（若需重生成 p1_features；提交包已含）
python scripts/run_p1.py
python scripts/check_p1_acceptance.py
python scripts/generate_p1_replay.py

# 问题二
python scripts/generate_p2_mask_library.py
python scripts/train_p2_ladder.py
python scripts/train_p2_mrfn.py
python scripts/train_p2_bft.py
python scripts/eval_p2_test.py
python scripts/eval_p2_test_second.py
python scripts/predict_fj3.py

# 问题三
python scripts/check_p3_m1.py
python scripts/check_p3_m2.py
python scripts/check_p3_m3.py
python scripts/check_p3_m4.py
python scripts/check_p3_m5.py
python scripts/gen_p3_m6_alignment.py
python scripts/predict_fj4.py
```

复现产物默认写入仓库根 `runs/` 目录；首次完整复现需约 30 GB 磁盘（含 `cache/` 中的 BERT/wav2vec 权重）。

## 六、可复现性说明

- **种子**：训练/评测/掩码库/对齐重生成 使用独立种子，记录于本 README 与各脚本常量
- **数据契约**：问题一特征契约 `p1.v2`、问题二评测掩码库种子（`MASK_LIBRARY_SEED = 2026`，`K_INSTANCES = 8`）见 [`src/p2/missing.py`](./src/p2/)
- **测试集封版**：唯一一次评估封版于 commit `03eece9`（论文附录 A.2），此后未再做测试集评估
- **权重指纹**：BERT `bert-base-uncased`（revision 86b5e09）与 MRFN+ 三种子检查点的 SHA-256 前缀记录于论文附录 A.2

---

**最后更新**：2026-09-27
