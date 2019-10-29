GPU_IMAGE?="tensortrade:latest-gpu"
CPU_IMAGE?="tensortrade:latest"

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

build-cpu: 
	docker build -t ${CPU_IMAGE} .

build-gpu:
	docker build -t ${GPU_IMAGE} . --build-arg gpu_tag="-gpu"

build-cpu-if-not-built: 
	if [ ! $$(docker images -q ${CPU_IMAGE}) ]; then $(MAKE) build-cpu; fi;

build-gpu-if-not-built: 
	if [ ! $$(docker images -q ${GPU_IMAGE}) ]; then $(MAKE) build-gpu; fi;

run-notebook: build-cpu-if-not-built
	docker run -it --rm -p=8888:8888 ${CPU_IMAGE} jupyter notebook --ip='*' --port=8888 --no-browser --allow-root ./examples/

run-docs: build-cpu-if-not-built
	if [ $$(docker ps -aq --filter name=tensortrade_docs) ]; then docker rm $$(docker ps -aq --filter name=tensortrade_docs); fi;
	docker run -t --name tensortrade_docs ${CPU_IMAGE} make docs-build && make docs-serve
	python3 -m webbrowser http://localhost:8000/docs/build/html/index.html

run-tests: build-cpu-if-not-built
	docker run -it --rm ${CPU_IMAGE} make test

run-notebook-gpu: build-gpu-if-not-built
	docker run -it --rm -p=8888:8888 ${GPU_IMAGE} jupyter notebook --ip='*' --port=8888 --no-browser --allow-root /examples/

run-docs-gpu: build-gpu-if-not-built
	if [ $$(docker ps -aq --filter name=tensortrade_docs) ]; then docker rm $$(docker ps -aq --filter name=tensortrade_docs); fi;
	docker run -t --name tensortrade_docs ${GPU_IMAGE} make docs-build && make docs-serve
	python3 -m webbrowser http://localhost:8000/docs/build/html/index.html

run-tests-gpu: build-gpu-if-not-built
	docker run -it --rm ${GPU_IMAGE} make test

package:
	rm -rf dist
	python3 setup.py sdist
	python3 setup.py bdist_wheel

test-release: package
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

release: package
	twine upload dist/*