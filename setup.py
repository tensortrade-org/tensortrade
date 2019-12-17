# Copyright 2019 The TensorTrade Authors.
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
    description='TensorTrade: a reinforcement learning library for training, evaluating, and deploying robust trading agents.',
    long_description='TensorTrade: a reinforcement learning library for training, evaluating, and deploying robust trading agents.',
    long_description_content_type='text/markdown',
    author='Adam King',
    author_email='adamjking3@gmail.com',
    url='https://github.com/notadamking/tensortrade',
    packages=[
        package for package in find_packages(exclude=('tests', 'docs'))
        if package.startswith('tensortrade')
    ],
    license='Apache 2.0',
    python_requires='>=3.6',
    install_requires=[
        'numpy==1.16.4',
        'pandas==0.25.0',
        'gym==0.14.0',
        'pyyaml==5.1.2',
        'sympy==1.4'
    ],
    extras_require={
        'tf': ['tensorflow==1.13.1'],
        'tf-gpu': ['tensorflow-gpu==1.13.1'],
        'tensorforce': ['tensorforce==0.5.2'],
        'baselines': ['stable-baselines==2.8.0', 'mpi4py'],
        'ta': ['ta==0.4.7'],
        'ta-lib': ['ta-lib==0.4.17'],
        'stochastic': ['stochastic==0.4.0'],
        'matplotlib': ['matplotlib==3.1.1'],
        'ccxt': ['ccxt==1.18.1220'],
        'all-framework': ['tensorflow==1.13.1',
                          'tensorforce==0.5.2',
                          'stable-baselines==2.8.0',
                          'mpi4py',
                          'ta==0.4.7',
                          'ta-lib==0.4.17',
                          'stochastic==0.4.0',
                          'ccxt==1.18.1220',
                          'matplotlib==3.1.1'
                          ],
        'tests': ['pytest==5.1.1'],
        'all-tests': ['tensorflow==1.13.1',
                      'tensorforce==0.5.2',
                      'ta==0.4.7',
                      'stochastic==0.4.0',
                      'ccxt==1.18.1220',
                      'pytest==5.1.1',
                      ],
        'docs': ['sphinx',
                 'sphinx_rtd_theme',
                 'sphinx_autodoc_typehints',
                 'sphinxcontrib.apidoc',
                 'nbsphinx',
                 'nbsphinx_link',
                 'm2r',
                 ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
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
