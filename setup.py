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


def split(sequence, sep):
    chunk = []
    for val in sequence:
        if val == sep:
            yield chunk
            chunk = []
        else:
            chunk.append(val)
    yield chunk


if sys.version_info.major != 3:
    raise NotImplementedError("TensorTrade is only compatible with Python 3.")


tensortrade_directory = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(tensortrade_directory, 'tensortrade', 'version.py'), 'r') as filehandle:
    for line in filehandle:
        if line.startswith('__version__'):
            version = line[15:-2]


with open(os.path.join(tensortrade_directory, 'requirements.txt'), encoding='utf-8') as f:
    install_requires = f.read().splitlines()

extras_require = {}

with open(os.path.join(tensortrade_directory, 'examples', 'requirements.txt'), encoding='utf-8') as f:
    extras_require['examples'] = f.read().splitlines()[2:]

with open(os.path.join(tensortrade_directory, 'requirements_extras.txt'), encoding='utf-8') as f:
    for extra in list(split(f.read().splitlines(), ''))[1:]:
        extras_require[extra[0][1:]] = extra[1:]

extras_require['full'] = \
    extras_require['ray'] + \
    extras_require['stable_baselines'] + \
    extras_require['tensorflow'] + \
    extras_require['pytorch'] + \
    extras_require['indicators'] + \
    extras_require['agents'] + \
    extras_require['binance'] + \
    extras_require['examples'] + \
    extras_require['tests'] + \
    extras_require['docs']


classifiers = [
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
]

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
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=classifiers,
    zip_safe=False
)

