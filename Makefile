PYTHON = python3
RESULTS = results

.PHONY: all experiments test paper clean

all: experiments test paper

experiments:
	$(PYTHON) -m src.experiments --results-dir $(RESULTS)

test:
	$(PYTHON) -m pytest tests/ -q

paper: experiments
	cd paper && pdflatex -interaction=nonstopmode paper09.tex
	cd paper && bibtex paper09
	cd paper && pdflatex -interaction=nonstopmode paper09.tex
	cd paper && pdflatex -interaction=nonstopmode paper09.tex

clean:
	rm -rf results/__pycache__ src/__pycache__ tests/__pycache__
	rm -f paper/*.aux paper/*.log paper/*.bbl paper/*.blg paper/*.out
