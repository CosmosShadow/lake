# coding: utf-8
from nose.tools import *
import lake


class TestMathFun(object):
	def test_PieceLinearFun(self):
		linear = lake.mt.PieceLinearFun(100, 200, 0, 1)
		assert_equal(linear(0), 0)
		assert_equal(linear(100), 0)
		assert_equal(linear(150), 0.5)
		assert_equal(linear(200), 1)
		assert_equal(linear(300), 1)