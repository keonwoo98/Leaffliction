.PHONY: setup lint format test smoke train verify clean

setup:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run flake8 src tests *.py

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest --cov=src/leaffliction

smoke:
	uv run pytest tests/test_smoke.py -v

train:
	uv run python train.py images/

verify:
	bash scripts/verify.sh

clean:
	rm -rf .venv .pytest_cache .ruff_cache build dist *.egg-info
