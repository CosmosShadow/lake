# coding: utf-8
import numpy as np


def one_hot(num, index):
	base = np.zeros(num)
	if 0 <= index < num:
		base[index] = 1
	return base


def zero_like(data):
	if hasattr(data, 'shape'):
		return np.zeros(data.shape)
	elif hasattr(data, '__iter__'):
		out = []
		for x in data:
			out.append(zero_like(x))
		return out
	else:
		return 0.


