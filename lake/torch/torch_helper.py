# coding: utf-8
import os
import sys
import json
import argparse
import lake.dir
import lake.file
import torch
import logging
import numpy as np
from recordtype import recordtype
import time
from . import network as torch_network
import numpy as np
from collections import defaultdict


class TorchHelper(object):
	def __init__(self, outputs_path = './outputs/', output=None, option_name=None):
		self._outputs_path = outputs_path
		self._output = output
		self._option_name = option_name
		self._load()

	def _parse_args(self):
		# 解析命令行输入
		parser = argparse.ArgumentParser()
		parser.add_argument('--option', type=str, default='', help='option')
		parser.add_argument('--output', type=str, default='', help='output')
		args, unknown = parser.parse_known_args()
		return args

	def _load(self):
		args = self._parse_args()
		self._load_output_dir(args)
		self._load_opt(args)

	@property
	def output_dir(self):
		return self._output_dir

	def _load_output_dir(self, args):
		# 确定输出目录，默认起一个时间
		# 命令行参数 > 传参 > 默认
		if len(args.output) > 0:
			self.output = args.output
		elif self._output is not None:
			self.output = self._output
		else:
			self.output = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

		self._output_dir  = os.path.join(self._outputs_path, self.output)
		lake.dir.mk(self._outputs_path)
		lake.dir.mk(self._output_dir)

	def _load_opt(self, args):
		"""加载option，顺序为:
		1、使用保存目录里的
		2、使用命令行指定的
		3、默认: option_base
		后两者需要保存到_output_dir目录下
		"""
		self._option_path = os.path.join(self._output_dir, 'option.json')
		if os.path.exists(self._option_path):
			option_json = lake.file.read(self._option_path)
			option_dict = json.loads(option_json)
			self.opt = recordtype('X', option_dict.keys())(*option_dict.values())
			print('从{}加载option'.format(self._option_path))
		else:
			# _option_name: 命令行 > 传参 > 默认
			if len(args.option) > 0:
				option_name = args.option
			elif self._option_name is not None:
				print(self._option_name)
				option_name = self._option_name
			else:
				option_name = 'base'
				
			sys.path.append('options')
			opt_pkg = __import__('option_' + option_name)
			self.opt = opt_pkg.Options()()
			self.opt.option_name = option_name
			option_json = json.dumps(vars(self.opt), indent=4)
			lake.file.write(option_json, self._option_path)
			print('从option_{}加载option'.format(option_name))

	def save_model(self, model, name):
		save_path = os.path.join(self._output_dir, '{}.pth'.format(name))
		if os.path.isfile(save_path):
			raise ValueError('{}模型已存在，不能覆盖'.format(save_path))
		# is_cuda = model.is_cuda
		torch.save(model.state_dict(), save_path)
		# if is_cuda:
			# model.cuda()

	def load_model(self, model, name=None):
		name = name or self.opt.model_load_name
		if name and len(name) > 0:
			model_path = os.path.json(self._output_dir, '%d.pth' % self.opt.model_load_name)
			if not os.path.isfile(model_path):
				raise ValueError('你想加载的模型%s不存在' % model_path)
			else:
				model.load_state_dict(torch.load(model_path))
				print('模型{}加载成功'.format(model_path))
		else:
			print('模型未加载')

	def default_optimizer(self, model):
		optimizer = torch.optim.Adam(model.parameters(), lr=self.opt.lr, weight_decay=self.opt.weight_decay)
		return optimizer


def _init_weight(m):
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
		if m.bias is not None:
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

def init_weight(model):
	self.apply(_init_weight)




