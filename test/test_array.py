# coding: utf-8
from nose.tools import *
import numpy as np
import lake


class TestArray(object):
	def test_sample(self):
		count = 100
		percent = 0.2

		data = range(count)
		sampled, left = lake.array.sample(data, percent)

		assert_equal(len(sampled), int(count * percent))
		assert_equal(len(left), int(count * (1-percent)))

	def test_exend(self):
		a = [1, 2, 3]
		assert_equal(len(lake.array.extend(a, 2)), 2)
		assert_equal(len(lake.array.extend(a, 4)), 4)
		assert_equal(len(lake.array.extend(a, 4, -1)), 4)
		assert_equal(lake.array.extend(a, 4, -1)[-1], -1)
		assert_equal(lake.array.extend(a, 4, -1)[0], 1)

	def test_flat(self):
		a = [[1, 2, 3], [4, 5]]
		assert_equal(len(lake.array.flat(a)), 5)

	def test_is_in(self):
		a = [[1, 2], [3, 4], [5, 6]]
		b = [1, 2]
		c = [3, 1]
		assert_true(lake.array.is_in(b, a))
		assert not lake.array.is_in(c, a)

	def test_split_with_length(self):
		data = range(100)
		datas = lake.array.split_with_length(data, 11)
		assert_equal(len(datas), 100/11 + 1)
		datas = lake.array.split_with_length(data, 11, fix_length=True)
		assert_equal(len(datas), 100/11)
		datas = lake.array.split_with_length(data, 11, step=6, fix_length=True)
		assert_equal(len(datas), (100 - 11)/6 + 1)
		datas = lake.array.split_with_length(data, 11, step=6)
		assert_equal(len(datas), 100/6 + 1)

	def test_shuffle(self):
		a = range(10)
		b = np.array(a)
		data = lake.array.Shuffler(a, b)
		for item in data(2):
			assert_equal(len(item[0]), 2)











