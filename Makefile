.PHONY: install test smoke benchmark clean

install:
	python -m pip install -e .

test:
	pytest

smoke:
	python scripts/train_baseline.py --config configs/smoke_cpu.yaml --router top2 --steps 10 --data synthetic --output experiments/results/smoke.json

benchmark:
	python benchmarks/dispatch_benchmark.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
