# coding: utf-8

class RingBuffer(object):
	def __init__(self, maxlen):
		self.maxlen = maxlen
		self.start = 0
		self.length = 0
		self.data = [None for _ in range(maxlen)]

	def __len__(self):
		return self.length

	def __getitem__(self, idx):
		if idx < 0 or idx >= self.length:
			raise KeyError()
		return self.data[(self.start + idx) % self.maxlen]

	def append(self, v):
		if self.length < self.maxlen:
			# We have space, simply increase the length.
			self.length += 1
		elif self.length == self.maxlen:
			# No space, "remove" the first item.
			self.start = (self.start + 1) % self.maxlen
		else:
			# This should never happen.
			raise RuntimeError()
		self.data[(self.start + self.length - 1) % self.maxlen] = v