GPU_IMAGE?="tensortrade:latest-gpu"
CPU_IMAGE?="tensortrade:latest"
SHM_SIZE?="3.0gb" # TODO: Automate me!

clean:
	find . | grep -E '(__pycache__|\.pyc|\.pyo$$)' | xargs rm -rf

test:
	pytest tests/

test-parallel:
	pytest --workers auto tests/

doctest:
	pytest --doctest-modules tensortrade/

docs-build:
	$(MAKE) -C docs html

docs-clean:
	$(MAKE) -C docs clean
	rm -rf docs/source/api

docs-serve:
	$(SHELL) -C cd docs/build/html
	python3 -m webbrowser http://localhost:8000/docs/build/html/index.html
	python3 -m http.server 8000

check-docker-auth:
	@docker pull --quiet python:3.12-slim > /dev/null 2>&1 || \
		(echo ""; \
		 echo "ERROR: Docker Hub authentication required."; \
		 echo "  Run 'docker login' first (free account: https://hub.docker.com/signup)"; \
		 echo ""; \
		 exit 1)

build-cpu: check-docker-auth
	docker build -t ${CPU_IMAGE} .

build-gpu: check-docker-auth
	docker build -t ${GPU_IMAGE} . --build-arg gpu_tag="-gpu"

build-cpu-if-not-built:
	if [ ! $$(docker images -q ${CPU_IMAGE}) ]; then $(MAKE) build-cpu; fi;

build-gpu-if-not-built:
	if [ ! $$(docker images -q ${GPU_IMAGE}) ]; then $(MAKE) build-gpu; fi;

run-notebook: build-cpu-if-not-built
	docker run -it --rm -p=8888:8888 -p=6006:6006 -v ${PWD}/examples:/app/examples -v ${PWD}/docs:/app/docs --shm-size=${SHM_SIZE} ${CPU_IMAGE} jupyter lab --ip='*' --port=8888 --no-browser --allow-root /app/

run-docs: build-cpu-if-not-built
	if [ $$(docker ps -aq --filter name=tensortrade_docs) ]; then docker rm $$(docker ps -aq --filter name=tensortrade_docs); fi;
	docker run -t --name tensortrade_docs --shm-size=${SHM_SIZE} ${CPU_IMAGE} make docs-build && make docs-serve
	python3 -m webbrowser http://localhost:8000/docs/build/html/index.html

run-tests: build-cpu-if-not-built
	docker run --rm --shm-size=${SHM_SIZE} ${CPU_IMAGE} pytest tests/ -m "not rllib"

run-notebook-gpu: build-gpu-if-not-built
	docker run -it --rm -p=8888:8888 -p=6006:6006 -v ${PWD}/examples:/app/examples -v ${PWD}/docs:/app/docs --shm-size=${SHM_SIZE} ${GPU_IMAGE} jupyter lab --ip='*' --port=8888 --no-browser --allow-root /app/

run-docs-gpu: build-gpu-if-not-built
	if [ $$(docker ps -aq --filter name=tensortrade_docs) ]; then docker rm $$(docker ps -aq --filter name=tensortrade_docs); fi;
	docker run -t --name tensortrade_docs --shm-size=${SHM_SIZE} ${GPU_IMAGE} make docs-build && make docs-serve
	python3 -m webbrowser http://localhost:8000/docs/build/html/index.html

run-tests-gpu: build-gpu-if-not-built
	docker run --rm --shm-size=${SHM_SIZE} ${GPU_IMAGE} pytest tests/ -m "not rllib"

package:
	rm -rf dist
	python3 setup.py sdist
	python3 setup.py bdist_wheel

test-release: package
	python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

release: package
	python3 -m twine upload dist/*
