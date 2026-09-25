论文图源目录。

结构图（scripts/figures/plot_diagrams.py，与 src/p2/models.py 契约逐条对应）：
- G1 总框架 g1_framework.{pdf,png}：P1→P2→P3 三列 + P1 映射表旁路到 P3；
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
