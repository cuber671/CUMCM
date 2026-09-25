# 论文大纲总控 v2

状态标记：☐ 未动笔 ｜ ◐ 有素材/部分成文 ｜ ☑ 成稿（待 LaTeX 化）｜ ✔ 定稿
规则：素材只指向已冻结方案与 runs/ 实测产物；数字进正文前与本表"必引数字"核对；
每章成文进 draft/，LaTeX 化进 latex/sections/；图表一律引用既有文件名，不重画。

## 全局口径（所有章节必须一致，写作前过一遍）

1. **模型名**：MRFN（P2 架构）→ MRFN+（R8 采纳版：BERT 解冻末层 + S_select 早停 + 3-seed 集成）；test 单模型 67.45% / 集成 68.64%，封版 MRFN 66.71%（同口径 +0.74pp）。
2. **两套反事实参照分名**：Shapley 用联盟级 v(∅)（avail≡0）；IG 用条件特征基线（avail=o）。不得混称。
3. **P3 保真度**：A 删除 19/20 胜（非 20/20）；F（text IG–LOO 0.170）如实写未通过 + 结构性诊断。
4. **附件3/4 无标签**：只做行为分析/解释展示，禁任何精度声明（锚点 5/5 仅内部自检）。
5. **P1 回放图**：签核完成前一律称"自动回放"。
6. **text 模态**：768 维为 BERT 上下文向量（逐 token 输出），池化+复制仅适用 audio/vision——与数据描述自洽。
7. **预注册纪律**：所有门槛运行前冻结；未过项不移动门槛（F、seed 4–5 扩充丢弃两例如实入文）。

## 章节映射与骨架

| 章 | 文件 | 状态 | 小节骨架 | 必引数字/图表 |
|---|---|---|---|---|
| 摘要 | 00-摘要.md | ☐ | 建模思路→三问方法→结果→创新点→关键词（≤2 页，最后写） | P1 100/100·cos=1.0；P2 67.45/68.64；P3 完备性 3e-8·A–E 过 |
| 一 问题重述 | 01-问题重述.md | ☐ | 1.1 背景 1.2 三问重述 1.3 附件清单（1–4） | 附件2 3395/728/727；附件3 30 条；附件4 20 条 |
| 二 问题分析与技术路线 | 02-问题分析与技术路线.md | ◐ | 2.1 三问耦合（P1 坐标系=P2/P3 接口）2.2 各问路线一句话 2.3 公共纪律（契约/指纹/预注册） | 图 g1_framework；闭环链路图 |
| 三 模型假设与符号说明 | 03-模型假设与符号说明.md | ☐ | 3.1 模型假设（半开区间、20ms 帧、CLS+48+SEP 截断、严格零行代理、无标签附件、四种缺失机制）3.2 符号表（o/b/a、v(S)、φ、IG、g、α、S_select、D_S、T=1.2142） | 方案 v2 §2/§4 + 两实施步骤文档 |
| 四 问题一 | 04-问题一.md | ◐ | 4.1 数据事实 4.2 统一坐标（语义网格/状态掩码/索引—时间映射）4.3 特征生成与边界规则 4.4 四指标验收 4.5 典型回放 | audio 非零率 99.915/99.732/100%；vision 94.473/94.293/93.925%；100/100·0 违例；7 条重叠逐位相等 cos=1.0；自生成 1.0/0.947；ok/review/rollback=43/33/24。图 q1_*×9+replay×9（自动回放）；表 q1_acceptance_summary |
| 五 问题二 | 05-问题二.md | ◐ | 5.1 数据契约与三态分账（掩码库 43×8×3）5.2 架构阶梯 B0→MRFN（掩码池化/缺失嵌入/可用性注意力/门控）5.3 训练（A35/B40/C15/D10、S_select）5.4 31 场景鲁棒性与三确定性结论 5.5 R7/R8 改进与采纳 5.6 附件3 交付 | 参数量 81k→417k；test 阶梯 p2_ladder；MRFN+ 67.45/68.64 vs 66.71；文本缺失主导脆弱性、a/v D_S≈0、帕累托前沿（R7）；valid S 0.7443。图 q2_*×4；表 p2_ladder/transfer/ablation/final_test |
| 六 问题三 | 06-问题三.md | **☑ 成稿** | 6.1 任务与两参照系 6.2 Shapley（完备性/哑玩家/锚点）6.3 IG（中点 64/词聚合/零泄漏）6.4 保真度 A–E 过+F 诊断 6.5 g–φ 6.6 附件4 交付 6.7 局限 | 完备性 3e-8/4e-16；40/40·0.62%；A 19/20·AOPC 0.166；F 0.170+a/v 0.739/0.732；g–φ text 0.716·argmax 90.7%；TOP3 100% 文本。图 q3_*×7；表 p3_*×6 |
| 七 实验与结果分析 | 07-实验与结果.md | ☐ | 7.1 统一实验协议（seed/指纹/门槛冻结）7.2 跨问汇总（阶梯+消融+保真度一张总表）7.3 错误归因（弱带、同文本跨标签不可约误差）7.4 计算成本 | 引用四五六图表为主，新增 1 张跨问汇总表（可选） |
| 八 模型评价与推广 | 08-模型评价与推广.md | ☐ | 8.1 优点（契约化管线/预注册纪律/缺失感知闭环）8.2 局限（P3 四条+P2 弱带与帕累托+P1 回放签核待办）8.3 推广（流式、跨域多模态、医学时序） | — |
| 参考文献 | 09-参考文献.md | ☐ | BERT/Shapley/IG(Sundararajan)/MOSEI/wav2vec2/ERASER/温度校准等，按引用次序编号 | latex/reference.bib 已有骨架 |
| 附录 | 10-附录.md | ☐ | A 环境与复现（environment-manifest）B 关键代码（align/mask_text_tokens/Shapley/IG）C 附件3/4 CSV 样例 D 掩码库说明 | cls 已含附录节 |

## 图表资产清单（已全部生成，正文按名引用）

- **总图**：g1_framework（三问闭环总架构）
- **P1（9 图 1 表）**：q1_pipeline / q1_unified_coordinate / q1_semantic_grid / q1_coordinate_mapping / q1_observation_heatmap / q1_quality_overview / q1_timing_coverage / q1_acceptance_summary / q1_behavior_detail + replay_×9（自动回放）；表 q1_acceptance_summary
- **P2（4 图 4 表）**：q2_mrfn_architecture / q2_degradation / q2_scenario_heatmap / q2_fj3_behavior；表 p2_ladder / p2_transfer / p2_ablation / p2_final_test
- **P3（7 图 6 表）**：q3_shapley_evidence / q3_ig_validation / q3_fidelity / q3_del_ins / q3_gphi / q3_att4 / q3_explain_cards；表 p3_completeness / p3_direction / p3_ig_validation / p3_fidelity / p3_gphi / p3_att4
- **待办**：P1 回放人工签核（runs/p1_replay/replay_audit.md）→ 签核后改称"人工核验回放"

## 写作顺序与依赖

一 → 二 → 三 → 四 → 五 →（六 ☑）→ 七 → 八 → 参考文献/附录 → **摘要（最后）** → LaTeX 接线（sections/*.tex → main.tex）→ xelatex 编译审计（格式清单见 latex/README.md）。
依赖说明：二需 g1_framework（已备）；七汇总依赖四五六成稿；摘要依赖全部。

## 工具链（已定）

draft/*.md（成文）→ latex/sections/*.tex（LaTeX 化）→ main.tex \input 接线 → `make`（xelatex+bibtex×2）。禁止 Pandoc 路线（已废弃）。图表脚本在 scripts/figures/（plot_q1*.py / plot_q2_results.py / plot_q3_results.py / plot_q3_cards.py，只读 runs/ 冻结产物）。

## 与其他目录的关系

docs/ = 冻结方案与实施记录（数字溯源）；runs/ = 实测产物（正文数字唯一来源）；src/ = 管线代码；data/、cache/、runs/ 大文件不入库（复现靠 manifest+seed+指纹）。
