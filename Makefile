PYTHON_BOOTSTRAP ?= python3.15
VENV_DIR ?= .venv-qa
PYTHON ?= $(VENV_DIR)/bin/python
NPM ?= npm

PYTHON_PATHS := scripts tests/python

.PHONY: bootstrap check check-deployed format-python install-browser install-dev install-node lint-python test-python

bootstrap: install-node install-browser install-dev

install-node:
	$(NPM) ci

install-browser: install-node
	./node_modules/.bin/playwright install chromium

install-dev:
	@if [ ! -e "$(VENV_DIR)" ]; then $(PYTHON_BOOTSTRAP) -m venv "$(VENV_DIR)"; fi
	$(PYTHON) -m pip install --disable-pip-version-check --requirement requirements-dev.txt

test-python:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m pytest --capture=sys -p no:cacheprovider

lint-python:
	$(PYTHON) -m flake8 $(PYTHON_PATHS)
	$(PYTHON) -m black --workers 1 --check --line-length 99 $(PYTHON_PATHS)

format-python:
	$(PYTHON) -m black --workers 1 --line-length 99 $(PYTHON_PATHS)

check:
	$(NPM) run check
	$(NPM) run test:browser
	$(MAKE) test-python
	$(MAKE) lint-python

check-deployed:
	@test -n "$(NIRU_DEPLOYED_BASE_URL)" || { echo "NIRU_DEPLOYED_BASE_URL is required" >&2; exit 2; }
	$(PYTHON) scripts/check_deployed_headers.py --base-url "$(NIRU_DEPLOYED_BASE_URL)"
