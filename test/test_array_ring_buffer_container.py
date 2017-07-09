# coding: utf-8
from nose.tools import *
import numpy as np
import lake


class TestRingBufferContainer(object):
	def test_basic(self):
		max_len = 20
		a = lake.array.RingBufferContainer(max_len, 3)
		for x in range(30):
			a.append([x, x, x])
		assert len(a) == 20

	def test_read_mask(self):
		max_len = 20
		a = lake.array.RingBufferContainer(max_len, 3, [1, 1, 2])
		for x in range(30):
			a.append([x, x, x])
		assert len(a) == 20

		count = 10
		values = a.next(count)
		assert len(values) == 3
		x, y, z = values
		assert len(x) == count
		assert len(y) == count
		assert len(z) == count
		assert x[0] == z[0][0]
		for i in range(count):
			assert isinstance(x[i], int)
			assert isinstance(y[i], int)
			assert len(z[i]) == 2

	def test_content(self):
		container = lake.array.RingBufferContainer(10, 4, [(-3, 5), 1, 1, 1])
		for x in xrange(1, 6):
			container.append([x]*4)
		a, b, c, d = container.next(1)
		assert_equal(a, [[1, 2, 3, 4, 5]])
		assert_equal(b, [4])
		assert_equal(c, [4])
		assert_equal(d, [4])


if __name__ == '__main__':
	TestRingBufferContainer().test_content()








