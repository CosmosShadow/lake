# coding: utf-8
from nose.tools import *
import lake.decorator


class TestDecorator(object):
	def test_register_fun(self):
		"""动态添加函数到类"""
		class A(object):
			def __init__(self):
				self._name = 'li'
			def fun(self):
				return self.hello()

		@lake.decorator.register_fun(A)
		def hello(self):
			return 'hello, ' + self._name

		assert_equal(A().fun(), 'hello, li')


	def test_duplicate_register(self):
		"""重复注册"""
		class A(object):
			def __init__(self):
				self._name = 'li'
			def fun(self):
				return self.hello()

		@lake.decorator.register_fun(A)
		def hello(self):
			return 'hello, ' + self._name

		@lake.decorator.register_fun(A)
		def hello(self):
			return 'hello again, ' + self._name

		assert_equal(A().fun(), 'hello again, li')