# coding: utf-8
from nose.tools import *
import lake


class TestDecorator(object):
	def test_register_fun(self):
		class A(object):
			def __init__(self):
				self._name = 'li'

			def fun(self):
				return self.hello()

		@lake.decorator.register_fun(A)
		def hello(self):
			return 'hello, ' + self._name

		assert_equal(A().fun(), 'hello, li')