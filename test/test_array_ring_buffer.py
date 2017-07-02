# coding: utf-8
import numpy as np
import lake


class TestRingBuffer(object):
	def test_ring_buffer(self):
		max_len = 20
		a = lake.array.RingBuffer(max_len)
		for x in range(30):
			a.append(x)
		assert len(a) == 20
		assert a[0] == 20