# coding: utf-8
import time
import singleton


def time_cost(func):
	def _time_cost(*args, **kwargs):
		start = time.time()
		ret = func(*args, **kwargs)
		print("==> time-cost: %6f     %s" % (time.time() - start, func.__name__))	#TODO: 颜色
		return ret
	return _time_cost


def dump_args(func):
	argnames = func.func_code.co_varnames[:func.func_code.co_argcount]
	fname = func.func_name
	def echo_func(*args,**kwargs):
		print fname + "(" + ', '.join(
			'%s=%r' % entry
			for entry in zip(argnames,args[:len(argnames)])+[("args",list(args[len(argnames):]))]+[("kwargs",kwargs)]) +")"
		return func(*args, **kwargs)
	return echo_func


if __name__ == '__main__':
	@time_cost
	@dump_args
	def print_hello(a):
		print 'hello', a
	print_hello({'hello': 'hello'})






