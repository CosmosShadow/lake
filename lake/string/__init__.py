# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function
from . import color

def humman(value):
	for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
		if abs(value) < 1024.0:
			return "%3.1f%s" % (value, unit)
		value /= 1024.0
	return "%.1f%s" % (value, 'Yi')