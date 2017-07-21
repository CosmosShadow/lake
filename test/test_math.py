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

	def test_log2(self):
		assert_equal(lake.mt.continue_divide(10, 2, max_left=2), (2, 2))
		assert_equal(lake.mt.continue_divide(8, 2, max_left=1), (3, 1))
		assert_equal(lake.mt.continue_divide(8, 2, max_left=2), (2, 2))
		assert_equal(lake.mt.continue_divide(8, 2, max_left=3), (2, 2))
		assert_equal(lake.mt.continue_divide(8, 2, max_left=4), (1, 4))
		assert_equal(lake.mt.continue_divide(7, 2, max_left=4), (1, 3))