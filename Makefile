GPU_IMAGE?="tensortrade:latest-gpu"
CPU_IMAGE?="tensortrade:latest"


docker-build:
	docker build -t tensortrade .

docker-run:
	docker run -ti -v examples -p 8888:8888 tensortrade

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
	$(SHELL) -c "cd docs/_build/html; python -m http.server 8000"

build-cpu: 
	docker build -t ${CPU_IMAGE} .

build-gpu:
	docker build -t ${GPU_IMAGE} . --build-arg gpu_tag="-gpu"

package:
	rm -rf dist
	python3 setup.py sdist
	python3 setup.py bdist_wheel

test-release: package
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

release: package
	twine upload dist/*
