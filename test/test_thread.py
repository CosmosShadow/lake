# coding: utf-8
from nose.tools import *
from lake.thread import pool_local
import time

class Test(object):
	def test_pool_local(self):
		def target(input_q, output_q, local):
			while not input_q.empty():
				data = input_q.get()
				output_q.put(data + local)
				time.sleep(0.1)

		locals = [2, 3, 5]
		inputs = range(100)
		outputs = pool_local(locals, target, inputs)
		assert_greater_equal(min(outputs), 2)
		assert_less_equal(max(outputs), 99 + 5)

if __name__ == '__main__':
	test = Test()
	test.test_pool_local()
