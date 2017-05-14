# coding: utf-8
import sys
# from builtins import object


class IteratorsPair(object):
	# 迭代器组合
	# 每次取datas的每一个data的一个元素，给成一个数组返回
	def __init__(self, datas, max_size=None):
		"""初始化
		Args:
			datas : Array of array, the size of each could be diffierent
			max_size: 最大长度，当datas中data有长度大于max_size时，停止迭代
		"""
		super(IteratorsPair, self).__init__()
		self.datas = datas
		self.stop_arr = [False] * len(datas)
		self.max_size = max_size


	def __iter__(self):
		self.stop_arr = [False] * len(self.datas)
		self.iter_arr = [iter(item) for item in self.datas]
		self.iter = 0
		return self


	def next(self):
		results = []
		for i, iterator in enumerate(self.iter_arr):
			item = None
			try:
				item = next(iterator)
			except StopIteration:
				self.stop_arr[i] = True
				self.iter_arr[i] = iter(self.datas[i])
				item = next(iterator)
			results.append(item)

		if all(self.stop_arr) or (self.max_size is not None and self.iter >= self.max_size):
			self.stop_arr = [False] * len(self.datas)
			raise StopIteration()
		else:
			self.iter += 1
			return results


	def __len__(self):
		if self.max_size is None:
			return max([len(item) for item in self.datas])
		else:
			return min(max([len(item) for item in self.datas]), self.max_size)












