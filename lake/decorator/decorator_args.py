# coding: utf-8

def dump_args(func):
	argnames = func.func_code.co_varnames[:func.func_code.co_argcount]
	fname = func.func_name
	def echo_func(*args,**kwargs):
		str = fname + "(" + ', '.join(
			'%s=%r' % entry
			for entry in zip(argnames,args[:len(argnames)])+[("args",list(args[len(argnames):]))]+[("kwargs",kwargs)]) +")"
		print(str)
		return func(*args, **kwargs)
	return echo_func