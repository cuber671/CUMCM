论文图源目录。

P3 图表（scripts/figures/plot_q3_cards.py，本会话版本，只读 runs/p3/{m2,m3,m4} 冻结产物）：
- q3_shapley_evidence（F1/F2 合并：带符号 φ+主导模态◆+TOP-8 词证据）/ q3_fidelity（F3 曲线，AOPC 0.1657 胜 19/20、ΔAUC 0.1291 胜 20/20）/ q3_ig_validation（F4：三基线 ρ̄≥0.96 + F 诊断 a/v 强一致 text 未达）/ q3_explain_cards（F6：#09/#14/#13，无精度声明）→ paper/latex/figures/q3/；
- 表 p3_shapley_direction / p3_completeness / p3_ig_validation / p3_fidelity → paper/latex/tables/（xelatex 0 Overfull）；
- 口径：40/40、max_rel 0.62%（64 步中点，6fa6c56）；A–E 通过、F text IG–LOO 0.1701<0.3 未通过+诊断行；A 胜数 19/20；v(∅) 与条件特征基线分开命名；附件4 无时间戳（词回溯=P1 网格，[t_s,t_e) 仅 P1 域内）；
- 图注要点（脚本 stdout）：φ 图按 phi_text 升序、vision 主导 #14/#19（18+0+2=20）；TOP-k 中同名词=不同 word_id 实例（#09 not×2 = id 3/11，#13 a×2 = id 5/16）；#13 哑玩家 φ_v≡0。
- **并行会话另有 scripts/figures/plot_q3_results.py**（产出 q3_del_ins/q3_gphi/q3_att4 + p3_shapley/p3_gphi/p3_att4，依赖 m5/m6 产物；截至本提交 tab_shapley 有 `max()` 对 float 的 bug 未跑通）。两脚本产物名不重叠，仅 p3_fidelity.tex 同名同义（本仓库版本含 F 诊断行，已验证）；q3_del_ins 与 q3_fidelity 内容重复（后者含胜数标注），正文二选一。

结构图（scripts/figures/plot_diagrams.py，与 src/p2/models.py 契约逐条对应）：
- G1 总框架 g1_framework.{pdf,png}：三泳道各五节点（输入→核心处理→输出）——P1 提取→CTC 对齐→区间映射→坐标资产；P2 = MRFN 三机制（缺失状态编码/可用性约束跨模态注意力/可靠性门控融合）→双头；P3 冻结预测器→Shapley+IG 归因→保真度验证→证据回溯；仅一级方法节点（工具/参数级细节不进图）；三条跨问题耦合边（坐标接口、模型+门控、P1 映射旁路）保留；
- P2-F1 q2_mrfn_architecture.{pdf,png}：左列缺失空间 (M,P,R,L)/a=o·(1−b)/text [MASK] 前置，右列 MRFN 七层数据流（§6.1 嵌入公式→BiGRU+6 方向注意力→池化/覆盖率→门控→双头）；
- 结构图公式一律用 mathtext（$...$），普通文本里禁止字面 h̃/ᵢ 等 combining 字符（Noto 缺字形）；结构图审计只查文字入盒/重叠，箭头触盒边是设计不算穿盒。

P2 图表（scripts/figures/plot_q2_results.py，全部只读冻结 json）：
- q2_degradation / q2_scenario_heatmap / q2_fj3_behavior → paper/latex/figures/q2/；
- p2_ladder / p2_transfer / p2_ablation / p2_final_test → paper/latex/tables/（xelatex 编译验证）。
- 口径纪律：主图/主表只用 runs/p2/test_final（B0–MRFN，附件2 test 唯一一次）；MRFN+ 仅进 p2_final_test（评估批次列区分「主表首次 test / 预注册追加 test」），不构造其鲁棒性曲线；单模型均值与 3-seed 集成分列不混比。
- 附件3 行为图如实报告：gate_text hi 0.616 < lo 0.638（Spearman −0.195）、text-only 一致率 0.600 < 0.867，**未观察到预期单调重分配/趋同**（fj3 json note 已同步修正）；极性分布用附件3 CSV 30 条样本口径（勿用 test_second 的 pred_dist——那是附件2 test 的 727 条）。

P1 图表：
- 图1「统一语义—时间坐标图」由 scripts/figures/plot_q1_unified.py 自动生成（样本 -3g5yACwYnA$_$13，VFR），输出 paper/latex/figures/q1/q1_unified_coordinate.{pdf,png}；帧 pts 缓存 runs/p1_replay/<safe>_frames_pts.json。**正文不再引用旧 q1_coordinate_mapping 与单样本 replay 图**（源文件保留供附录与复现）；replay 集人工签核（runs/p1_replay/replay_audit.json 的 human_conclusion）未完成前，正文表述用「自动回放」，不得写成已完成人工验证。
- 图2「50 步语义网格与状态掩码」由 scripts/figures/plot_q1.py::plot_semantic_grid 从冻结 pkl 自动生成（样本 -a55Q6RWvTA$_$3，头部截断长样本），输出至 paper/latex/figures/q1/q1_semantic_grid.{pdf,png}。
- 图3+表「验收一表一图」由 plot_q1.py::plot_acceptance_summary / write_acceptance_table 自动生成：图 paper/latex/figures/q1/q1_acceptance_summary.{pdf,png}（Panel A 对齐质量状态 43/33/24 / Panel B 观测率分布 / Panel C vision 缺失原因构成，分母=content-position 级 N=2247）；表 paper/latex/tables/q1_acceptance_summary.tex（booktabs，xelatex 编译验证通过）。**图注措辞：review/rollback 是对齐质量状态、非管线失败**；Panel B 标题长度接近面板宽度上限，加长需先查跨 panel 重叠。
- 两图的 position 域与 time 域坐标独立（图1 仅以映射梯形连接），不得把 WordPiece 位置线性拉伸到物理时间轴——「非 50 等间隔帧」结论由图2 时间映射行承担。
- manifest_paper.csv 的「对齐状态」列对齐 manifest.csv 的 alignment_status（43/33/24），「处理状态」是管线 status（全 ok），二者不可混用。
