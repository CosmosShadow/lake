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


def long_substr(data):
	"""
	data数组包含的所有字符串中，查找最长相似项
	"""
	substr = ''
	if len(data) > 1 and len(data[0]) > 0:
		for i in range(len(data[0])):
			for j in range(len(data[0])-i+1):
				if j > len(substr) and is_substr(data[0][i:i+j], data):
					substr = data[0][i:i+j]
	return substr

def is_substr(find, data):
	if len(data) < 1 and len(find) < 1:
		return False
	for i in range(len(data)):
		if find not in data[i]:
			return False
	return True