from setuptools import setup
from setuptools import find_packages

long_description = '''
A reinforcement learning library for training, evaluating, and deploying robust trading agents with TF2.
'''

setup(name='TensorTrade',
      version='2.2.4',
      description='A reinforcement learning library for training, evaluating, and deploying robust trading agents with TF2.',
      long_description=long_description,
      author='Adam King',
      author_email='adamjking3@gmail.com',
      url='https://github.com/notadamking/tensortrade',
      download_url='https://github.com/notadamking/tensortrade/tarball/0.0.1',
      license='GPLv3',
      install_requires=[],
      extras_require={
          'tests': [],
      },
      classifiers=[
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GPLv3 License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())
