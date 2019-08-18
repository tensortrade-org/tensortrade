GPU_IMAGE?="tensortrade:latest-gpu"
CPU_IMAGE?="tensortrade:latest"

.PHONY: yapf lint clean sync lock test test-parallel docs-build docs-clean circle build-cpu build-gpu release

yapf:
	yapf -vv -ir .
	isort -y

lint:
	flake8 .
	pydocstyle .
	mypy .

clean:
	find . | grep -E '(__pycache__|\.pyc|\.pyo$$)' | xargs rm -rf

sync:
	pipenv sync --dev

lock:
	pipenv lock --dev 

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

circle:
	circleci config validate
	circleci local execute --job build

build-cpu: 
	docker build -t ${CPU_IMAGE} .

build-gpu:
	docker build -t ${GPU_IMAGE} . --build-arg gpu_tag="-gpu"

package:
	python setup.py sdist
	python setup.py bdist_wheel

test-release: package
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

release: package
	twine upload dist/*