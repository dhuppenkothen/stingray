#!/usr/bin/env python

import glob
import os
import sys

from setuptools import setup


setup(name='stingray',
      version='0.1.dev',
      description='Time Series Methods For Astronomical X-ray Data',
      author='Stingray Developers',
      author_email='spectraltiming-stingray@googlegroups.com',
      license='MIT',
      url='https://github.com/StingraySoftware/stingray',
      packages=['stingray'],
      classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2.6',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: Implementation :: CPython',
          'Topic :: Scientific/Engineering :: Astronomy',
          'Topic :: Scientific/Engineering :: Physics'
      ],
      test_suite='nose.collector',
)
