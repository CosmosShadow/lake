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


def encode32(value):
	"""转成32位码
	Args:
		value : string
	Returns:
		value : string
	"""
	if isinstance(value, unicode):
		value = value.encode('utf-8')
	import binascii
	from hashids import Hashids
	hashids = Hashids(alphabet='0123456789abcdefghijklmnopqrstuvwxyz', min_length=6)
	md5 = lambda s: hashlib.md5(s).hexdigest()
	crc32 = lambda s: '%08X' % (binascii.crc32(s) & 0xffffffff)
	return crc32(value)


def md5(txt):
	import hashlib
	md5_obj = hashlib.md5()
	md5_obj.update(txt)
	hash_code = md5_obj.hexdigest()
	md5_str = str(hash_code).lower()
	return md5_str
