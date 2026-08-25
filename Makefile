# Convenience wrapper. `uv run ...` is the canonical interface and works
# identically on Windows, macOS and Linux; this file exists for Unix habit
# and for anyone who types `make` before reading the README.

.PHONY: help install lint fmt types imports toc toc-check test check backtest bound clean

help:
	@echo "install  create the environment from uv.lock"
	@echo "lint     ruff check"
	@echo "fmt      ruff format (writes)"
	@echo "types    mypy"
	@echo "imports  import-linter: no solver outside model/"
	@echo "toc      regenerate the index at the top of every document"
	@echo "test     pytest"
	@echo "check    lint + types + imports + toc + test, as CI runs it"
	@echo "backtest every policy over both regimes, all three c_deg values"
	@echo "bound    the annual-window bound (DECISIONS.md 2.4)"

install:
	uv sync --all-groups --locked

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff format .

types:
	uv run mypy

imports:
	uv run lint-imports

test:
	uv run pytest -q

# Private notes are globbed, not listed: their filenames are nobody's
# business but this working copy's, and a clone simply has none.
DOCS := README.md CLAUDE.md docs/DECISIONS.md data/README.md hpc/README.md $(wildcard docs/private/*.md)

# Indexes are generated, never hand-maintained: a renamed section silently
# breaks its own anchor otherwise. The private notes are gitignored, so the
# tool skips whatever a given clone does not have.
toc:
	uv run python tools/toc.py $(DOCS)

toc-check:
	uv run python tools/toc.py --check $(DOCS)

check: lint types imports toc-check test

# The headline is the quarter-hourly regime; the hourly one is historical
# context and the two are not comparable (DECISIONS.md 1.1).
backtest:
	uv run bess-arb run --regime quarter_hourly --sweep --json results/backtest_quarter_hourly.json
	uv run bess-arb run --regime hourly --sweep --json results/backtest_hourly.json

bound:
	uv run bess-arb bound --regime hourly --json results/annual_bound.json

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache
