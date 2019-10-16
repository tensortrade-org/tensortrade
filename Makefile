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

run-notebook: 
	docker run -t --rm -p=8888:8888 ${CPU_IMAGE} jupyter notebook --ip='*' --port=8888 --no-browser --allow-root /examples/

run-docs: 
	docker run -t --name documentation ${CPU_IMAGE} make docs-build
	docker cp documentation:/docs/. ./docs/
	docker container rm documentation

run-test: 
	docker run -it --rm ${CPU_IMAGE} make test

package:
	rm -rf dist
	python3 setup.py sdist
	python3 setup.py bdist_wheel

test-release: package
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

release: package
	twine upload dist/*
