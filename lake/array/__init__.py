# coding: utf-8
from shuffler import Shuffler


def extend(values, count):
	values = values if type(values) is list else [values]
	if len(values) >= count:
		return values[:count]
	else:
		return values + [values[-1]] * (count - len(values))