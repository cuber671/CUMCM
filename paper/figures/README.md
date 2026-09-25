论文图源目录。
- 图1「统一语义—时间坐标图」由 scripts/figures/plot_q1_unified.py 自动生成（样本 -3g5yACwYnA$_$13，VFR），输出 paper/latex/figures/q1/q1_unified_coordinate.{pdf,png}；帧 pts 缓存 runs/p1_replay/<safe>_frames_pts.json。**正文不再引用旧 q1_coordinate_mapping 与单样本 replay 图**（源文件保留供附录与复现）；replay 集人工签核（runs/p1_replay/replay_audit.json 的 human_conclusion）未完成前，正文表述用「自动回放」，不得写成已完成人工验证。
- 图2「50 步语义网格与状态掩码」由 scripts/figures/plot_q1.py::plot_semantic_grid 从冻结 pkl 自动生成（样本 -a55Q6RWvTA$_$3，头部截断长样本），输出至 paper/latex/figures/q1/q1_semantic_grid.{pdf,png}。
- 两图的 position 域与 time 域坐标独立（图1 仅以映射梯形连接），不得把 WordPiece 位置线性拉伸到物理时间轴——「非 50 等间隔帧」结论由图2 时间映射行承担。
- manifest_paper.csv 的「对齐状态」列对齐 manifest.csv 的 alignment_status（43/33/24），「处理状态」是管线 status（全 ok），二者不可混用。
