.PHONY: test lint format build clean

test:
	pytest

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

build:
	python -m build

clean:
	rm -rf build dist .pytest_cache .ruff_cache src/*.egg-info
