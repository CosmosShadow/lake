# coding: utf-8
from nose.tools import *
import lake.string


class Test(object):
	def test_md5(self):
		txt = 'hello world'
		md5 = lake.string.md5(txt)
		assert_equal(md5, '5eb63bbbe01eeed093cb22bb8f5acdc3')

if __name__ == '__main__':
	test = Test()
	test.test_md5()