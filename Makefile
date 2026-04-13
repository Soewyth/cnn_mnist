.PHONY: setup test lint train clean

setup:
	python -m venv .venv && \
	.venv/bin/pip install -r requirements.txt && \
	.venv/bin/pip install -e .

test:
	.venv/bin/python -m pytest tests/ -q

lint: 
	.venv/bin/ruff check . && .venv/bin/ruff format --check .

train:
	.venv/bin/python -m cnn_mnist.train

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
	rm -rf outputs/models/* outputs/figures/*  