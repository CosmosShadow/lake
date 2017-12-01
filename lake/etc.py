# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import collections


def dumps(obj, nested_level=0, output=sys.stdout):
	"""格式化输出，主要解决utf-8编码无法看的问题"""
	spacing = '\t'
	if isinstance(obj, (dict, collections.OrderedDict)):
		print('%s{' % ((nested_level) * spacing), file=output)
		for k, v in obj.items():
			if hasattr(v, '__iter__'):
				print('%s%s:' % ((nested_level + 1) * spacing, k), file=output)
				dumps(v, nested_level + 1, output)
			else:
				print('%s%s: %s' % ((nested_level + 1) * spacing, k, v), file=output)
		print('%s}' % (nested_level * spacing), file=output)
	elif type(obj) == list:
		print('%s[' % ((nested_level) * spacing), file=output)
		for v in obj:
			if hasattr(v, '__iter__'):
				dumps(v, nested_level + 1, output)
			else:
				print('%s%s' % ((nested_level + 1) * spacing, v), file=output)
		print('%s]' % ((nested_level) * spacing), file=output)
	else:
		print('%s%s' % (nested_level * spacing, obj), file=output)