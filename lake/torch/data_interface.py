# coding: utf-8
# created by lichen(lichenarthurdata@gmail.com)
# 数据接口

class DataInterface(object):

	def next(self, batch_size):
		raise NotImplementedError()

	def count(self, batch_size):
		raise NotImplementedError()
