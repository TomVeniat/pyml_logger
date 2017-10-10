#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup,find_packages

setup(name='pyml_logger',
      version='1.0.2',
      description='A simple set of Logging classes for experiments',
      author='Ludovic Denoyer',
      author_email='ludovic.denoyer@lip6.fr',
      url='https://github.com/ludc/pyml_logger',
      packages=find_packages()
     )
