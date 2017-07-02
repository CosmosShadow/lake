# coding: utf-8

class PieceLinearFun(object):
	"""分段线性函数"""
	def __init__(self, step1, step2, value1, value2):
		super(PieceLinearFun, self).__init__()
		assert step1 < step2
		self.step1 = step1
		self.step2 = step2
		self.value1 = value1
		self.value2 = value2

	def __call__(self, step):
		if step < self.step1:
			return self.value1
		elif step > self.step2:
			return self.value2
		else:
			scale = float(self.value2 - self.value1) / (self.step2 - self.step1)
			return self.value1 + scale * (step - self.step1)