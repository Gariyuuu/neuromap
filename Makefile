.PHONY: setup data features-smoke encode-smoke benchmark rsa compression analyze figures paper site test

VENV = .venv/bin/python3

setup:
	python3.11 -m venv .venv
	$(VENV) -m pip install --upgrade pip
	$(VENV) -m pip install -r requirements.txt

data:
	$(VENV) scripts/01_fetch_neural_data.py

features-smoke:
	$(VENV) scripts/02_extract_features.py

encode-smoke:
	$(VENV) -c "import sys; sys.path.insert(0,'.'); from neuromap.encoding import run_encoding; import numpy as np; \
	X=np.random.randn(20,10); Y=np.random.randn(20,5); r=run_encoding(X,Y,n_outer_folds=4,n_pca_components=5); \
	print('smoke test mean_r:', r.mean_r)"

benchmark:
	$(VENV) scripts/03_run_encoding.py

rsa:
	$(VENV) scripts/04_run_rsa.py

compression:
	$(VENV) scripts/05_run_compression.py

analyze:
	$(VENV) scripts/06_ranking_stability.py
	$(VENV) scripts/07_failure_analysis.py

figures:
	$(VENV) scripts/08_make_figures.py

site:
	$(VENV) scripts/09_build_site_data.py

paper:
	@echo "See paper/paper.md — rendered manually or via pandoc"

test:
	$(VENV) -m pytest tests/ -v
