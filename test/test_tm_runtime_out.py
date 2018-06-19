# coding: utf-8
# created by lichen(lichenarthurdata@gmail.com)
# 测试运行超时

from nose.tools import *
from lake.tm import TimeOut, TimeOutError
import time

class Test(object):
	def test_runtime_out(self):	
		success = False
		try:
			with TimeOut(1) as t:
				time.sleep(2)
		except TimeOutError as e:
			success = True
		assert_true(success)


if __name__ == '__main__':
	test = Test()
	test.test_runtime_out()