try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')
requirements = open('requirements.txt').read().splitlines()
test_requirements = requirements + ['flake8']

setup(name='hax',
      version='0.2',
      description="Handy Analysis for XENON",
      long_description=readme + '\n\n' + history,
      url='https://github.com/XENON1T/hax',
      license='MIT',
      package_data={'hax': ['runs_info/*.csv', 'pax_classes/*.cpp', 'minitrees', 'hax.ini']},
      package_dir={'hax': 'hax'},
      packages=['hax',
                'hax.treemakers'],
      scripts=['bin/haxer'],
      py_modules=['hax'],
      install_requires=requirements,
      classifiers=['Intended Audience :: Developers',
                   'Development Status :: 3 - Alpha',
                   'Programming Language :: Python :: 3'],
      zip_safe=False)
