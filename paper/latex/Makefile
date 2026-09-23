# “华为杯”第二十三届 论文编译（必须 xelatex）
MAIN = main
all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex
	xelatex -interaction=nonstopmode $(MAIN).tex
	bibtex $(MAIN)
	xelatex -interaction=nonstopmode $(MAIN).tex
	xelatex -interaction=nonstopmode $(MAIN).tex

clean:
	rm -f *.aux *.log *.toc *.out *.bbl *.blg *.fls *.fdb_latexmk *.xdv
.PHONY: all clean
