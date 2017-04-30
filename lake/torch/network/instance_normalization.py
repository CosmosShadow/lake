# coding: utf-8
import torch
import torch.nn as nn

class InstanceNormalization(nn.Module):
	def __init__(self, dim, eps=1e-5):
		super(InstanceNormalization, self).__init__()
		self.weight = nn.Parameter(torch.FloatTensor(dim))
		self.bias = nn.Parameter(torch.FloatTensor(dim))
		self.eps = eps
		self._reset_parameters()

	def _reset_parameters(self):
		self.weight.data.uniform_()
		self.bias.data.zero_()

	def forward(self, x):
		n = x.size(2) * x.size(3)
		t = x.view(x.size(0), x.size(1), n)
		mean = torch.mean(t, 2).unsqueeze(2).expand_as(x)
		var = torch.var(t, 2).unsqueeze(2).expand_as(x) * ((n - 1) / float(n))
		scale_broadcast = self.weight.unsqueeze(1).unsqueeze(1).unsqueeze(0)
		scale_broadcast = scale_broadcast.expand_as(x)
		shift_broadcast = self.bias.unsqueeze(1).unsqueeze(1).unsqueeze(0)
		shift_broadcast = shift_broadcast.expand_as(x)
		out = (x - mean) / torch.sqrt(var + self.eps)
		out = out * scale_broadcast + shift_broadcast
		return out