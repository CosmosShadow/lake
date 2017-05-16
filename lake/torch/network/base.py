# coding: utf-8
import torch
import torch.nn as nn
import lake.torch

class Base(nn.Module):
	def __init__(self, gpu_ids=None):
		super(Base, self).__init__()
		self.gpu_ids = gpu_ids or lake.torch.network.get_default_gpu_ids()
		self.model = None
		if len(self.gpu_ids) > 0:
			assert(torch.cuda.is_available())

	def init_weights(self):
		if len(self.gpu_ids) > 0:
			self.cuda(device_id=self.gpu_ids[0])
		self.apply(self._init_weight)

	def print_net(self):
		print(self)
		params_count = sum([param.numel() for param in self.parameters()])
		print('Total number of parameters: %s' % lake.string.humman(params_count))

	def _init_weight(self, m):
		classname = m.__class__.__name__
		if classname.find('Conv') != -1:
			m.weight.data.normal_(0.0, 0.02)
		elif classname.find('BatchNorm2d') != -1 or  classname.find('InstanceNorm2d') != -1:
			print m
			m.weight.data.normal_(1.0, 0.02)
			m.bias.data.fill_(0)

	def forward(self, input):
		assert self.model is not None
		if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
			return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
		else:
			return self.model(input)

	def save_network(self, save_path):
		torch.save(self.cpu().state_dict(), save_path)
		if len(self.gpu_ids) > 0 and torch.cuda.is_available():
			self.cuda(device_id=self.gpu_ids[0])

	def load_network(self, save_path):
		self.load_state_dict(torch.load(save_path))






















