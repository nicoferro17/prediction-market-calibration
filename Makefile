.PHONY: install lint format typecheck test check

install:
	uv sync --dev

lint:
	uv run ruff check src tests

format:
	uv run ruff format --check src tests

typecheck:
	uv run mypy src tests

test:
	uv run pytest -q

check: lint format typecheck test
