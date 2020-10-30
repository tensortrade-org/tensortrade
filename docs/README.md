# Documentation for TensorTrade

Read [the documentation](https://tensortrade.readthedocs.io).

This directory contains the sources (`.md` and `.rst` files) for the
documentation. The main index page is defined in `source/index.rst`.
The Sphinx options and plugins are found in the `source/conf.py` file.
The documentation is generated in full by calling `make html` which
also automatically generates the Python API documentation from
docstrings.

## Building documentation locally

Dependencies must be installed using `make sync` from the project root.
Run `make docs-build` from project root, or `make html` from the `docs/` subfolder (this one).

Note this can take some time as some of the notebooks may be executed
during the build process. The resulting documentation is located in the
`build` directory with `build/html/index.html` marking the homepage.

## Sphinx extensions and plugins

We use various Sphinx extensions and plugins to build the documentation:

- [recommonmark](https://recommonmark.readthedocs.io) - to handle both `.rst` and `.md`
- [sphinx.ext.napoleon](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) - support extracting Numpy style doctrings for API doc generation
- [sphinx_autodoc_typehints](https://github.com/agronholm/sphinx-autodoc-typehints) - support parsing of typehints for API doc generation
- [sphinxcontrib.apidoc](https://github.com/sphinx-contrib/apidoc) - automatic running of [sphinx-apidoc](https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html) during the build to document API
- [nbsphinx](https://nbsphinx.readthedocs.io) - parsing Jupyter notebooks to generate static documentation
- [nbsphinx_link](https://nbsphinx-link.readthedocs.io) - support linking to notebooks outside of Sphinx source directory via `.nblink` files

The full list of plugins and their options can be found in `source/conf.py`.
