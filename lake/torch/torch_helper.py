# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function
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
	def __init__(self, outputs_path = './outputs/', output=None, option_name=None, log_to_console=False, epoch_to_load=None):
		"""description
		Args:
			log_to_console: 显示到命令行，有其它模块设置了logging
		"""
		self.log_to_console = log_to_console
		self._outputs_path = outputs_path
		self._option_name = option_name
		self._epoch_to_load = epoch_to_load
		self._output = output
		self.data_train = None
		self.data_test = None
		self._model = None
		self.optimizer = None
		self.hooks = []
		self._load()
		self._reset_record()

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

		self.record_path = os.path.join(self._output_dir, 'record.txt')

		self._set_gpu()
		self._config_logging()
		self._load_epoch()

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

	def _set_gpu(self):
		if torch.cuda.is_available():
			torch_network.set_default_gpu_ids([int(item) for item in self.opt.gpu_ids])

	def _config_logging(self):
		log_path = self._output_dir + 'train.log'
		format = '%(asctime)s - %(levelname)s - %(name)s[line:%(lineno)d]: %(message)s'

		# 文件记录
		logging.basicConfig(
				filename = log_path,
				filemode = 'a',
				level = logging.INFO,
				format = format)

		# 控制台输出
		if self.log_to_console:
			root = logging.getLogger()
			ch = logging.StreamHandler(sys.stdout)
			ch.setLevel(logging.INFO)
			formatter = logging.Formatter(format)
			ch.setFormatter(formatter)
			root.addHandler(ch)

		self._logger = logging.getLogger(__name__)

	@property
	def model(self):
		return self._model

	def init_weight(self, model):
		init_weight(model)

	@model.setter
	def model(self, value):
		self._model = value
		self.init_weight(self._model)
		self._model.output_dir = self._output_dir
		try:
			if self._epoch_to_load is not None:
				model_path = os.path.join(self._output_dir, '%d.pth' % self._epoch_to_load)
				if not os.path.isfile(model_path):
					raise ValueError('你想加载的模型%s不存在' % model_path)
			else:
				model_path = self.last_model_path()
			if model_path is not None:
				print(model_path)
				self._model.load_state_dict(torch.load(model_path))
				self._logger.info('模型{}加载成功'.format(model_path))
			else:
				self._logger.info('模型未加载')
		except Exception as e:
			print(e)
			self._logger.info('模型加载出错')
			self.epoch = 1

	def _load_epoch(self):
		self.epoch = 1
		if os.path.exists(self.record_path):
			records = lake.file.read(self.record_path)
			if len(records) > 0 and len(records[-1].strip()) > 0:
				self.epoch = int(json.loads(records[-1])['epoch'])

	def default_optimizer(self):
		optimizer = torch.optim.Adam(
				self._model.parameters(),
				lr=self.opt.lr,
				weight_decay=self.opt.weight_decay)
		return optimizer

	def _reset_record(self):
		self._epoch_records = defaultdict(list)
		self._epoch_start = time.time()

	def add_record(self, key, value):
		self._epoch_records[key].append(value)

	def add_records(self, records):
		for key, value in records.items():
			self.add_record(key, value)

	def _epoch_log(self, values):
		results = ['epoch: %d' % self.epoch]
		for key, value in values.items():
			if key != 'epoch':
				value = ('%.6f' % value) if isinstance(value, float) else str(value)
				results.append('%s: %s' % (key, value))
		self._logger.info('   '.join(results))

	def _store_record(self):
		if self.epoch % self.opt.print_per == 0:
			rdf = lambda x: round(x, 6)
			values = {}
			for key, value in self._epoch_records.items():
				if isinstance(value[0], (int, float)):
					value = rdf(np.mean(value))
				else:
					value = value[-1]
				values[key] = value
				values['epoch'] = self.epoch
			if hasattr(self, 'current_lr'):
				values['lr'] = self.current_lr
			values['time'] = rdf(time.time() - self._epoch_start)
			self._epoch_log(values)
			record_json = json.dumps(values)
			lake.file.add_line(record_json, self.record_path)
			self._reset_record()


	def new_model_path(self):
		path = os.path.join(self._output_dir, '%d.pth' % self.epoch)
		if os.path.isfile(path):
			raise ValueError('模型已经存在，不能覆盖保存')
		return path

	def last_model_path(self):
		paths = lake.dir.loop(self._output_dir, ['.pth'])
		if len(paths) > 0:
			epochs = np.array([int(os.path.basename(x).split('.')[0]) for x in paths])
			index = np.argmax(epochs)
			path = paths[index]
			return path
		else:
			return None

	def step(self):
		if self.epoch % self.opt.save_per == 0:
			torch.save(self._model.state_dict(), self.new_model_path())
			self.add_record('save', 1)
		self._store_record()
		self.epoch += 1

	def train_stop(self):
		self._model.save_network(self.new_model_path())
		self._logger.info('train finish')

	def finished(self):
		return self.epoch >= self.opt.epochs


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
	model.apply(_init_weight)








