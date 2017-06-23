# coding: utf-8

class CountNextInterface(object):
	def __init__(self):
		super(CountNextInterface, self).__init__()

	def next(self, batch_size):
		raise NotImplementedError("not implemented in base calss")

	def count(self, batch_size=None):
		raise NotImplementedError("not implemented in base calss")