.PHONY: setup test lint bench compare clean

setup:
	pip install -e ".[dev,plot]"

test:
	pytest

lint:
	ruff check src/ tests/

bench:
	cachepilot bench --policy perc --workload mixed --requests 1000

compare:
	cachepilot compare --workload mixed --requests 2000 --spike 500

bench-full:
	python scripts/run_bench.py benchmarks/mixed_spike.yaml --out results/mixed_spike.json
	python scripts/run_bench.py benchmarks/memory_pressure.yaml --out results/memory_pressure.json

plot:
	python scripts/plot_results.py results/mixed_spike.json --out results/mixed_spike.png

clean:
	rm -rf results/ __pycache__ .pytest_cache src/*.egg-info
