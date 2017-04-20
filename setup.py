#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
	name = 'lake',
	version = '0.0.1',
	keywords = ('lake'),
	description = 'a python lib for network, file, and something else',
	license = 'MIT License',

	url = 'https://github.com/CosmosShadow/lake',
	author = 'lichen',
	author_email = 'lichenarthurdata@gmail.com',

	packages = find_packages(),
	include_package_data = True,
	platforms = 'any',
	install_requires = [],
)