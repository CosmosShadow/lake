# coding: utf-8
import itertools
import random
import numpy as np
from collections import *


def to_dict(data):
	"""数组转字典，字典元素为数组
	Args:
		data: [(k, v), (k, v), ...]
	Returns:
		b : type
	"""
	result = defaultdict(list)
	for k, v in data:
		result[k].append(v)
	return result


def to_raw_dict(data):
	"""数组转字典，字典元素为数组
	Args:
		data: [arr, arr, ...]
	Returns:
		{arr[0]: [arr, arr, ...], ...}
	"""
	result = defaultdict(list)
	for item in data:
		result[item[0]].append(item)
	return result

def extend(values, count, value=None):
	"""扩展
	Args:
		values: Array
		count: int
	Returns:
		values_extended: values长度不及count时，用最后一个元素补齐; 长度多时，截断
	"""
	values = values if type(values) is list else [values]
	if len(values) >= count:
		return values[:count]
	else:
		value = value or values[-1]
		return values + [value] * (count - len(values))


def flat(arr_arr):
	"""压平: 数组的数组 -> 数组
	Args:
		arr_arr: Array of Array
	Returns:
		arr: Array
	"""
	return list(itertools.chain.from_iterable(arr_arr))


def is_in(arr, arr_arr):
	"""check array is in array of array
	Args:
		arr: check value
		arr_arr: check container
	"""
	for the_arr in arr_arr:
		if len(arr) == len(the_arr):
			if all([var1==var2 for var1, var2 in zip(arr, the_arr)]):
				return True
	return False


def split_with_length(data, length, step=None, fix_length=False):
	step = step or length
	arr = []
	for index in range(0, len(data), step):
		if fix_length:
			if index + length <= len(data):
				arr.append(data[index: index+length])
		else:
			stop_index = min(index + length, len(data))
			arr.append(data[index: stop_index])
	return arr


def categorize(arr, index=0):
	# 根据第index个元素，分类数组的数组
	assert isinstance(arr, Iterable)
	results = defaultdict(list)
	for record in arr:
		assert 0 <= index < len(record)
		results[record[index]].append(record)
	return results

