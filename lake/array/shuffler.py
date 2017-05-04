# coding: utf-8
import numpy as np

class Shuffler(object):
	def __init__(self, *datas):
		super(Shuffler, self).__init__()
		self.datas = list(datas)
		self.length = self.obj_size(self.datas[0])
		for i in range(1, len(self.datas)):
			assert self.obj_size(self.datas[i]) == self.length

	def obj_size(self, obj):
		return len(obj) if isinstance(obj, list) else obj.size(0)

	def __len__():
		return self.length

	def __call__(self, batch_size):
		assert batch_size <= self.length

		# shuffle原始数据
		perm = np.arange(self.length)
		np.random.shuffle(perm)
		for i in range(len(self.datas)):
			self.datas[i] = [self.datas[i][j] for j in perm] if isinstance(self.datas[i], list) else self.datas[i][perm]

		for i in range(0, self.length+1-batch_size, batch_size):
			yield (data[i:i+batch_size] for data in self.data)