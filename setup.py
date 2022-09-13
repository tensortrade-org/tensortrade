# Copyright 2021 The TensorTrade Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License


import sys
import os

from setuptools import find_packages, setup


if sys.version_info.major != 3:
    raise NotImplementedError("TensorTrade is only compatible with Python 3.")


tensortrade_directory = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(tensortrade_directory, 'tensortrade', 'version.py'), 'r') as filehandle:
    for line in filehandle:
        if line.startswith('__version__'):
            version = line[15:-2]

setup(
    name='tensortrade',
    version=version,
    description='TensorTrade: A reinforcement learning library for training, evaluating, and deploying robust trading agents.',
    long_description='TensorTrade: A reinforcement learning library for training, evaluating, and deploying robust trading agents.',
    long_description_content_type='text/markdown',
    author='Adam King <adamjking3@gmail.com>, Matthew Brulhardt <mwbrulhardt@gmail.com>',
    maintainer='Carlo Grisetti <carlo.grisetti@gmail.com>',
    url='https://github.com/tensortrade-org/tensortrade',
    packages=[
        package for package in find_packages(exclude=('tests', 'docs'))
        if package.startswith('tensortrade')
    ],
    license='Apache 2.0',
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.17.0',
        'pandas>=0.25.0',
        'gym>=0.15.7',
        'pyyaml>=5.1.2',
        'stochastic>=0.6.0',
        'tensorflow>=2.7.0',
        'ipython>=7.12.0',
        'matplotlib>=3.1.1',
        'plotly>=4.5.0',
        'deprecated>=1.2.13'
    ],
    extras_require={
        'tests': [
            'pytest>=5.1.1',
            'ta>=0.4.7'
        ],
        'docs': [
            'sphinx',
            'sphinx_rtd_theme',
            'sphinxcontrib.apidoc',
            'nbsphinx',
            'nbsphinx_link',
            'recommonmark',
            'sphinx_markdown_tables',
            'ipykernel'
        ],
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Natural Language :: English',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Office/Business :: Financial :: Investment',
        'Topic :: Office/Business :: Financial',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: System :: Distributed Computing',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    zip_safe=False
)
