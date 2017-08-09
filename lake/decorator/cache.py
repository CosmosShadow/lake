# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import sys
if sys.version > '3':
	from functools import lru_cache
else:
	from repoze.lru import lru_cache