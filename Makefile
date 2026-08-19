# Convenience wrapper. `uv run ...` is the canonical interface and works
# identically on Windows, macOS and Linux; this file exists for Unix habit
# and for anyone who types `make` before reading the README.

.PHONY: help install lint fmt types test check clean

help:
	@echo "install  create the environment from uv.lock"
	@echo "lint     ruff check"
	@echo "fmt      ruff format (writes)"
	@echo "types    mypy"
	@echo "test     pytest"
	@echo "check    lint + types + test, as CI runs it"

install:
	uv sync --all-groups --locked

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff format .

types:
	uv run mypy

test:
	uv run pytest -q

check: lint types test

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache
