.PHONY: install lint format typecheck test collect check

install:
	uv sync --dev

lint:
	uv run ruff check src tests scripts

format:
	uv run ruff format --check src tests scripts

typecheck:
	uv run mypy src tests scripts

test:
	uv run pytest -q

collect:
	uv run python scripts/collect_daily.py --dry-run

check: lint format typecheck test
