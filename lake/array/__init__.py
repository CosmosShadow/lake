# coding: utf-8
from shuffler import Shuffler
from iterator_pair import IteratorsPair

import itertools


def extend(values, count):
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
		return values + [values[-1]] * (count - len(values))


def flat(arr_arr):
	"""压平: 数组的数组 -> 数组
	Args:
		arr_arr: Array of Array
	Returns:
		arr: Array
	"""
	return list(itertools.chain.from_iterable(arr_arr))