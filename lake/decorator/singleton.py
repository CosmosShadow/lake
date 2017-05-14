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
		if cls not in instances:
			instances[cls] = cls(*args, **kwargs)
		return instances[cls]
	return _singleton

