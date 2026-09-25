论文图源目录。
- Figure 1 总架构图规格见 docs/问题一建模方案.md §7；三轨对齐图由 P1 管线对齐表自动生成。
- 图2「50 步语义网格与状态掩码」由 scripts/figures/plot_q1.py::plot_semantic_grid 从冻结 pkl 自动生成（样本 -a55Q6RWvTA$_$3，头部截断长样本），输出至 paper/latex/figures/q1/q1_semantic_grid.{pdf,png}。
- manifest_paper.csv 的「对齐状态」列对齐 manifest.csv 的 alignment_status（43/33/24），「处理状态」是管线 status（全 ok），二者不可混用。
