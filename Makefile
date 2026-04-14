.PHONY: setup test lint train clean

PYTHON := $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null)

ifeq ($(strip $(PYTHON)),)
    $(error No Python interpreter found. Please install Python 3: https://www.python.org/downloads/)
endif

setup:
	@$(PYTHON) -c "import sys; sys.exit(0) if sys.version_info >= (3,10) else sys.exit(1)" || \
		(echo "Error: Python 3.10+ required, found $$($(PYTHON) --version)"; exit 1)
	@echo "Using $$($(PYTHON) --version) at $(PYTHON)"
	$(PYTHON) -m venv .venv && \
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