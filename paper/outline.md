# 论文大纲总控 v3（2026-09-25 冻结）

状态标记：☐ 未动笔 ｜ ◐ 有素材/部分成文 ｜ ☑ 成稿（待 LaTeX 化）｜ ✔ 定稿
规则：素材只指向已冻结方案与 runs/ 实测产物；数字进正文前与本表"必引数字"核对；
每章成文进 draft/，LaTeX 化进 latex/sections/；图表一律引用既有文件名，不重画。

v3 变更（相对 v2）：
- 七章合并结构冻结：原"七 实验与结果分析"+"八 模型评价与推广"合并为
  "七 模型综合评价与推广"；每问章内直接完成"方法—实验—结果—题面回应"，
  第七章只做跨问题综合分析，不重复汇报指标。
- 3.1 假设编号 = **H1–H10**（对齐 docs/实验假设与验证目标.md 实际表；
  H10=文本缺失无泄漏。不写 H1–H8/H1–H9，防编号漂移）。
- 4.4 标题改为"对齐求解算法：CTC 强制对齐与失败分层处理"（正文说明
  插值放行、质量标记、人工核查清单）。
- 5.5 标题改为"模型改进、采纳与追加终评"（MRFN=主表冻结正式模型；
  MRFN+=valid 预注册筛选后采纳的补充变体；追加 test 不替换主表、无
  31 场景鲁棒性曲线）。
- 6.4 标题改为"解释可信性验证：A–E 通过与 F 项结构性诊断"。
- 7.2 用"三问联合验证：门控—信息量—贡献关系及其边界"（不写"三角互证"：
  a/v g–φ 较弱 + text IG–LOO 存在反事实语义差异，"互证"过强）。
- 全篇图号由 LaTeX 顺序编号：G1 总框架图=图 1，其后按章顺排；caption 与
  正文一律用 \ref，不手写图号。
- P3 正文图表定版（4 图 4 表）：q3_shapley_evidence / q3_fidelity /
  q3_ig_validation / q3_explain_cards；q3_del_ins 退出正文（与 q3_fidelity
  重复，后者含 A 胜 19/20 标注）；q3_gphi / q3_att4 及表 p3_gphi /
  p3_att4 / p3_shapley（并行会话产物，p3_shapley 因脚本 bug 未生成）
  一律不纳入正文正式目录，是否进附录待定。
- 正文正式表合计 10 张：一×1（附件总览）+ P1×1 + P2×4 + P3×4。

## 全局口径（所有章节必须一致，写作前过一遍）

1. **模型名**：MRFN（P2 架构）→ MRFN+（R8 采纳版：BERT 解冻末层 + S_select 早停 + 3-seed 集成）；test 单模型 67.45% / 集成 68.64%，封版 MRFN 66.71%（同口径 +0.74pp）。
2. **两套反事实参照分名**：Shapley 用联盟级 v(∅)（avail≡0）；IG 用条件特征基线（avail=o）。不得混称。
3. **P3 保真度**：A 删除 19/20 胜（非 20/20）；F（text IG–LOO 0.170）如实写未通过 + 结构性诊断。
4. **附件3/4 无标签**：只做行为分析/解释展示，禁任何精度声明（锚点 5/5 仅内部自检）。
5. **P1 回放图**：签核完成前一律称"自动回放"。
6. **text 模态**：768 维为 BERT 上下文向量（逐 token 输出），池化+复制仅适用 audio/vision——与数据描述自洽。
7. **预注册纪律**：所有门槛运行前冻结；未过项不移动门槛（F、seed 4–5 扩充丢弃两例如实入文）。

## 章节映射与骨架（七章结构）

| 章 | 文件 | 状态 | 小节骨架 | 必引数字/图表 |
|---|---|---|---|---|
| 摘要 | 00-摘要.md | ☐ | 建模思路→三问方法→结果→创新点→关键词（≤2 页，最后写） | P1 100/100·cos=1.0；P2 67.45/68.64；P3 完备性 3e-8·A–E 过 |
| 一 问题重述 | 01-问题重述.md | ✔ | 1.1 背景 1.2–1.4 三问重述 1.5 附件说明 | 附件2 3395/728/727；附件3 30 条；附件4 20 条 |
| 二 问题分析与总体技术路线 | 02-问题分析与技术路线.md | ◐ | 2.1 三问耦合（P1 坐标系=P2/P3 接口）2.2 各问路线一句话 2.3 公共纪律（契约/指纹/预注册） | **图1** g1_framework |
| 三 模型假设与符号说明 | 03-模型假设与符号说明.md | ☐ | 3.1 模型假设 **H1–H10**（半开区间、20ms 帧、CLS+48+SEP 截断、严格零行代理、无标签附件、四种缺失机制、文本缺失无泄漏）3.2 符号表（o/b/a、v(S)、φ、IG、g、α、S_select、D_S、T=1.2142） | docs/实验假设与验证目标.md |
| 四 问题一：多模态时序对齐模型的建立与求解 | 04-问题一.md | ◐ | 4.1 数据事实 4.2 统一坐标（语义网格/状态掩码/索引—时间映射）4.3 特征生成与边界规则 4.4 **对齐求解算法：CTC 强制对齐与失败分层处理**（插值放行/质量标记/人工核查清单）4.5 四指标验收与对题面回应 4.6 典型回放（自动回放） | audio 非零率 vs 99.915%；vision vs 94.473%；100/100·0 违例；7 条重叠逐位相等 cos=1.0；ok/review/rollback=43/33/24。图 q1_unified_coordinate / q1_semantic_grid / q1_acceptance_summary；表1 q1_acceptance_summary |
| 五 问题二：缺失场景下鲁棒情感预测模型的建立与求解 | 05-问题二.md | ◐ | 5.1 数据契约与三态分账 5.2 架构阶梯 B0→MRFN（掩码池化/缺失嵌入/可用性注意力/门控）5.3 训练（A35/B40/C15/D10、S_select）5.4 31 场景鲁棒性与三确定性结论 5.5 **模型改进、采纳与追加终评**（MRFN 主表冻结 / MRFN+ 追加不替换主表、无鲁棒性曲线）5.6 附件3 交付 | 参数量 81k→417k；MRFN clean S 0.7561·D_S 0.0283·text80 0.618 vs B3 0.557；MRFN+ 67.45/68.64 vs 66.71。图 q2_mrfn_architecture / q2_degradation / q2_scenario_heatmap / q2_fj3_behavior；表 p2_ladder / transfer / ablation / final_test |
| 六 问题三：可解释情感预测模型的建立与求解 | 06-问题三.md | **☑ 成稿** | 6.1 任务与两参照系 6.2 Shapley（完备性/哑玩家/锚点）6.3 IG（中点 64/词聚合/零泄漏）6.4 **解释可信性验证：A–E 通过与 F 项结构性诊断** 6.5 g–φ 6.6 附件4 交付与解释卡 6.7 局限 | 完备性 3e-8/4e-16；40/40·0.62%（6fa6c56）；A 19/20·AOPC 0.166；F 0.170+a/v 0.739/0.732；g–φ text 0.716。图 q3_shapley_evidence / q3_fidelity / q3_ig_validation / q3_explain_cards；表 p3_shapley_direction / completeness / ig_validation / fidelity |
| 七 模型综合评价与推广 | 07-综合评价.md（原 07/08 合并） | ☐ | 7.1 优点（契约化管线/预注册纪律/缺失感知闭环）7.2 **三问联合验证：门控—信息量—贡献关系及其边界**（a/v g–φ 弱、text IG–LOO 语义差异如实写）7.3 局限（P3 四条+P2 弱带+P1 回放签核待办）7.4 错误归因（同文本跨标签不可约误差）7.5 推广（流式/跨域多模态/医学时序） | 不重复分问指标；引用四五六 \ref |
| 参考文献 | 09-参考文献.md | ☐ | BERT/Shapley/IG(Sundararajan)/MOSEI/wav2vec2/ERASER/温度校准 | latex/reference.bib |
| 附录 | 10-附录.md | ☐ | A 环境与复现 B 关键代码 C 附件3/4 CSV 样例 D 掩码库说明 E replay 自动回放图 | replay_×9；q3_del_ins/q3_gphi/q3_att4 备选 |

## 图表资产清单（正文正式 = 12 图 10 表；编号由 LaTeX 顺排）

- **一**：表 attachments（附件总览，1.5 节）
- **图1** g1_framework（第二章）
- **P1**：q1_unified_coordinate / q1_semantic_grid / q1_acceptance_summary + 表 q1_acceptance_summary
- **P2**：q2_mrfn_architecture / q2_degradation / q2_scenario_heatmap / q2_fj3_behavior + 表 p2_ladder / transfer / ablation / final_test
- **P3**：q3_shapley_evidence / q3_fidelity / q3_ig_validation / q3_explain_cards + 表 p3_shapley_direction / completeness / ig_validation / fidelity
- **附录/备选（不进正文）**：replay_×9、q1 旧图（coordinate_mapping/observation_heatmap/quality_overview/timing_coverage/pipeline/behavior_detail）、q3_del_ins、q3_gphi、q3_att4、p3_gphi、p3_att4、p3_shapley（未生成）
- **待办**：P1 回放人工签核（runs/p1_replay/replay_audit.md）→ 签核后改称"人工核验回放"

## 写作顺序与依赖

一 → 二 → 三 → 四 → 五 →（六 ☑）→ 七 → 参考文献/附录 → **摘要（最后）**。
LaTeX 接线：main.tex 已含七章树与全部 12 图 9 表的 figure/table 环境（caption+label 齐备，图号 \ref 自动）；各章正文成文后逐段替换素材注释。

## 工具链（已定）

draft/*.md（成文）→ latex/sections/*.tex（LaTeX 化）→ main.tex \input 接线 → `make`（xelatex+bibtex×2）。图表脚本在 scripts/figures/（plot_q1*.py / plot_q2_results.py / plot_q3_cards.py / plot_diagrams.py，只读 runs/ 冻结产物；plot_q3_results.py 为并行会话版本，其正文产物未纳入）。

## 与其他目录的关系

docs/ = 冻结方案与实施记录（数字溯源）；runs/ = 实测产物（正文数字唯一来源）；src/ = 管线代码；data/、cache/、runs/ 大文件不入库（复现靠 manifest+seed+指纹）。
