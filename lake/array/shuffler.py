# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from .count_next_interface import *

class Shuffler(CountNextInterface):
	def __init__(self, *datas):
		super(Shuffler, self).__init__()
		self.datas = list(datas)
		self.length = self.obj_size(self.datas[0])
		for i in range(1, len(self.datas)):
			assert self.obj_size(self.datas[i]) == self.length
		self.index = 0

	def obj_size(self, obj):
		return len(obj) if isinstance(obj, list) else obj.shape[0]

	def __len__():
		return self.length

	def test(self, batch_size):
		assert batch_size <= self.length
		return (data[0:batch_size] for data in self.datas)

	def _shuffle(self):
		perm = np.arange(self.length)
		np.random.shuffle(perm)
		for i in range(len(self.datas)):
			self.datas[i] = [self.datas[i][j] for j in perm] if isinstance(self.datas[i], list) else self.datas[i][perm]

	def __call__(self, batch_size, is_fix=True):
		assert batch_size <= self.length
		self._shuffle()
		if is_fix:
			for i in range(0, self.length+1-batch_size, batch_size):
				yield tuple([data[i: i+batch_size] for data in self.datas])
		else:
			for i in range(0, self.length, batch_size):
				yield tuple([data[i: min(i+batch_size, len(data))] for data in self.datas])

	def next(self, batch_size):
		assert batch_size <= self.length
		if self.index + batch_size > self.length:
			self.index = 0
			self._shuffle()
		self.index = self.index + batch_size
		return tuple([data[self.index-batch_size: self.index] for data in self.datas])

	def count(self, batch_size=None):
		if batch_size is None:
			return self.length
		else:
			return self.length / batch_size
