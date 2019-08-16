from setuptools import setup
from setuptools import find_packages

long_description = """
A reinforcement learning library for training, evaluating, and deploying robust trading agents with TF2.
"""

setup(name='tensortrade',
      version='0.0.1a1',
      description='A reinforcement learning library for training, evaluating, and deploying robust trading agents with TF2.',
      long_description=long_description,
      author='Adam King',
      author_email='adamjking3@gmail.com',
      url='https://github.com/notadamking/tensortrade',
      download_url='https://github.com/notadamking/tensortrade/tarball/0.0.1',
      license='Apache 2.0',
      install_requires=[
          'numpy',
          'pandas',
          'sklearn',
          'gym',
          'gin-config',
          'ccxt',
          'stochastic',
          'hyperopt'
      ],
      extras_require={
          'tests': ['pytest'],
      },
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
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
      packages=find_packages())
