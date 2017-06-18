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

	@property
	def use_cuda(self):
		return self.gpu_ids is not None and len(self.gpu_ids) > 0 and torch.cuda.is_available()

	@property
	def multi_gpu(self):
		return self.use_cuda and len(self.gpu_ids) > 1

	def init_weights(self):
		if self.use_cuda:
			self.cuda(device_id=self.gpu_ids[0])
		self.apply(self._init_weight)

	def print_net(self):
		print(self)
		params_count = sum([param.numel() for param in self.parameters()])
		print('Total number of parameters: %s' % lake.string.humman(params_count))

	def _init_weight(self, m):
		classname = m.__class__.__name__
		if classname.find('BatchNorm2d') != -1 or  classname.find('InstanceNorm2d') != -1:
			m.weight.data.normal_(1.0, 0.02)
			m.bias.data.fill_(0)
		elif classname.find('Conv') != -1:
			weight_shape = list(m.weight.data.size())
			fan_in = np.prod(weight_shape[1:4])
			fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
			w_bound = np.sqrt(6. / (fan_in + fan_out))
			m.weight.data.uniform_(-w_bound, w_bound)
			m.bias.data.fill_(0)
		elif classname.find('Linear') != -1:
			weight_shape = list(m.weight.data.size())
			fan_in = weight_shape[1]
			fan_out = weight_shape[0]
			w_bound = np.sqrt(6. / (fan_in + fan_out))
			m.weight.data.uniform_(-w_bound, w_bound)
			m.bias.data.fill_(0)
		else:
			pass

	def forward(self, input):
		assert self.model is not None
		if self.multi_gpu and isinstance(input.data, torch.cuda.FloatTensor):
			return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
		else:
			return self.model(input)

	def save_network(self, save_path):
		torch.save(self.cpu().state_dict(), save_path)
		if self.use_cuda:
			self.cuda(device_id=self.gpu_ids[0])

	def load_network(self, save_path):
		self.load_state_dict(torch.load(save_path))






















