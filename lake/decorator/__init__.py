# coding: utf-8
import time

import singleton

def time_cost(func):
	def _time_cost(*args, **kwargs):
		start = time.time()
		ret = func(*args, **kwargs)
		print("%s time cost: %.5f" func.__name__, time.time() - start)
		return ret
	return _time_cost