# coding: utf-8
from shuffler import Shuffler
from ring_buffer import RingBuffer
from iterator_pair import IteratorsPair
import itertools
import random
import numpy as np


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


def sample(data, sample_rate):
	"""
	Args:
		data : list or numpy
		sample_rate : 
	Returns:
		sampled, left
	"""
	count = len(data)
	perm = np.arange(count)
	np.random.shuffle(perm)

	sample_count = int(count * sample_rate) if isinstance(sample_rate, float) else sample_rate

	assert sample_count > 0
	assert sample_count < count

	sample_index = perm[:sample_count]
	left_index = perm[sample_count:]

	if isinstance(data, list):
		return [data[i] for i in sample_index], [data[i] for i in left_index]
	else:
		return data[sample_index], data[left_index]


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






