# coding: utf-8
import time
import lake


def time_cost(func):
	def _time_cost(*args, **kwargs):
		start = time.time()
		ret = func(*args, **kwargs)
		print("==> time-cost: %6f     %s" % (time.time() - start, func.__name__))
		return ret
	return _time_cost


def time_cost_red(func):
	def _time_cost(*args, **kwargs):
		start = time.time()
		ret = func(*args, **kwargs)
		print(lake.string.color.red("time: %6f ms    %s" % ((time.time() - start)*1000, func.__name__)))
		return ret
	return _time_cost