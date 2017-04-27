# coding: utf-8
import torch
import torch.nn as nn

class Base(nn.Module):
	def __init__(self, gpu_ids=[]):
		super(Base, self).__init__()
		self.gpu_ids = gpu_ids
		self.model = None

	def forward(self, input):
		assert self.model is not None
		if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
			return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
		else:
			return self.model(input)