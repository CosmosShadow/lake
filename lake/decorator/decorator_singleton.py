# coding: utf-8

def singleton(cls, *args, **kwargs):
	"""
	单例装饰器
	使用方法:
	@singleton
	class MyClass(object):
		a = 1
		def __init__(self, x=0):
			self.x = x

	one = MyClass()
	two = MyClass()
	assert id(one) == id(two)
	"""
	instances = {}
	def _singleton(*args, **kwargs):
		key = cls.__name__ + "(" + ', '.join('%s=%r' % entry for entry in [("args", list(args)), ("kwargs", kwargs)]) +")"
		if key not in instances:
			instances[key] = cls(*args, **kwargs)
		return instances[key]
	return _singleton

