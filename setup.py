import os
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')
requirements = open('requirements.txt').read().splitlines()
test_requirements = requirements + ['flake8']


if os.environ.get('READTHEDOCS', None) == 'True':
    # Remove all C/complicated libraries from requirements when we're in readthedocs
    # See http://docs.readthedocs.io/en/latest/faq.html#i-get-import-errors-on-libraries-that-depend-on-c-modules
    requirements = [q for q in requirements
                    if q not in 'numpy pandas numba dask'.split()]

setup(name='hax',
      version='2.4.0',
      description="Handy Analysis for XENON",
      long_description=readme + '\n\n' + history,
      url='https://github.com/XENON1T/hax',
      license='MIT',
      package_data={'hax': ['runs_info/*.csv', 'pax_classes/*.cpp', 'minitrees', 'hax.ini', 'sc_variables.csv']},
      package_dir={'hax': 'hax'},
      packages=['hax',
                'hax.treemakers'],
      scripts=['bin/haxer'],
      py_modules=['hax'],
      install_requires=requirements,
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: End Users/Desktop',
          'Programming Language :: Python :: 3'
      ],
      zip_safe=False)
