GPU_IMAGE?="tensortrade:latest-gpu"
CPU_IMAGE?="tensortrade:latest"
SHM_SIZE?="3.0gb"

.PHONY: clean install install-dev install-all sync lock test test-parallel doctest docs-build docs-clean docs-serve lint format typecheck build-cpu build-gpu build-cpu-if-not-built build-gpu-if-not-built run-notebook run-docs run-tests run-notebook-gpu run-docs-gpu run-tests-gpu package test-release release

# Development setup
install:
	uv sync

install-test:
	uv sync --group test

install-lint:
	uv sync --group lint

install-docs:
	uv sync --group docs

install-dev:
	uv sync --group dev

install-all:
	uv sync --all-extras --group dev

sync:
	uv sync --all-extras --group dev

lock:
	uv lock

# Cleaning
clean:
	find . | grep -E '(__pycache__|\.pyc|\.pyo$$)' | xargs rm -rf
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov dist build *.egg-info

# Testing
test:
	uv run pytest tests/

test-parallel:
	uv run pytest --workers auto tests/

doctest:
	uv run pytest --doctest-modules tensortrade/

# Linting and formatting
lint:
	uv run ruff check tensortrade/ tests/

format:
	uv run ruff format tensortrade/ tests/
	uv run ruff check --fix tensortrade/ tests/

typecheck:
	uv run ty check tensortrade/

# Documentation
docs-build:
	$(MAKE) -C docs html

docs-clean:
	$(MAKE) -C docs clean
	rm -rf docs/source/api

docs-serve:
	cd docs/build/html && python3 -m http.server 8000

# Docker builds
build-cpu:
	docker build -t ${CPU_IMAGE} .

build-gpu:
	docker build -t ${GPU_IMAGE} . --build-arg gpu_tag="-gpu"

build-cpu-if-not-built:
	if [ ! $$(docker images -q ${CPU_IMAGE}) ]; then $(MAKE) build-cpu; fi;

build-gpu-if-not-built:
	if [ ! $$(docker images -q ${GPU_IMAGE}) ]; then $(MAKE) build-gpu; fi;

# Docker run commands
run-notebook: build-cpu-if-not-built
	docker run -it --rm -p=8888:8888 -p=6006:6006 -v ${PWD}/examples:/app/examples -v ${PWD}/docs:/app/docs --shm-size=${SHM_SIZE} ${CPU_IMAGE} jupyter lab --ip='*' --port=8888 --no-browser --allow-root /app/

run-docs: build-cpu-if-not-built
	if [ $$(docker ps -aq --filter name=tensortrade_docs) ]; then docker rm $$(docker ps -aq --filter name=tensortrade_docs); fi;
	docker run -t --name tensortrade_docs --shm-size=${SHM_SIZE} ${CPU_IMAGE} make docs-build && make docs-serve
	python3 -m webbrowser http://localhost:8000/docs/build/html/index.html

run-tests: build-cpu-if-not-built
	docker run --rm --shm-size=${SHM_SIZE} ${CPU_IMAGE} pytest tests/

run-notebook-gpu: build-gpu-if-not-built
	docker run -it --rm -p=8888:8888 -p=6006:6006 -v ${PWD}/examples:/app/examples -v ${PWD}/docs:/app/docs --shm-size=${SHM_SIZE} ${GPU_IMAGE} jupyter lab --ip='*' --port=8888 --no-browser --allow-root /app/

run-docs-gpu: build-gpu-if-not-built
	if [ $$(docker ps -aq --filter name=tensortrade_docs) ]; then docker rm $$(docker ps -aq --filter name=tensortrade_docs); fi;
	docker run -t --name tensortrade_docs --shm-size=${SHM_SIZE} ${GPU_IMAGE} make docs-build && make docs-serve
	python3 -m webbrowser http://localhost:8000/docs/build/html/index.html

run-tests-gpu: build-gpu-if-not-built
	docker run --rm --shm-size=${SHM_SIZE} ${GPU_IMAGE} pytest tests/

# Publishing
package:
	rm -rf dist
	uv build

test-release: package
	uv publish --publish-url https://test.pypi.org/legacy/

release: package
	uv publish
