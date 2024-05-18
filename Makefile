SOURCE_DIR = src
PYTHON_FILES = `find . -path './build' -prune -o -path './dist' -prune -o -name '*.py' -print`;


# version checks
.PHONY: tool-check
tool-check:
	pip install -U black
	pip install -U isort
	pip install -U flake8
	pip install -U pylint


# code formating
.PHONY: format
format:
	black $(PYTHON_FILES)
	isort $(PYTHON_FILES)


# linter
.PHONY: lint
lint:
	isort --profile black --check-only $(PYTHON_FILES)
	pylint $(PYTHON_FILES)
	flake8 --max-line-length 120 $(PYTHON_FILES)


# test
.PHONY: test
test:
	pytest -rP


# format, lint and test
.PHONY: all
all:
	make format
	make lint
	make test
