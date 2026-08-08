.PHONY: install test smoke audit clean

install:
	python -m pip install -e .

test:
	PYTHONPATH=src pytest -q

audit:
	PYTHONPATH=src python scripts/audit_paper_reference.py

smoke:
	PYTHONPATH=src python scripts/run_s1_aliasing.py --n 5000 --out outputs/s1.json
	PYTHONPATH=src python scripts/train_synthetic.py --config configs/synthetic_tiny.yaml --outdir runs/tiny

clean:
	rm -rf runs outputs .pytest_cache
