install:
	uv pip install -r requirements.txt

format:
	black src tests
	isort src tests

lint:
	flake8 src tests

test:
	pytest --cov=src --cov-report=term-missing

typecheck:
	mypy src tests

docs:
	sphinx-build -b html docs docs/_build 