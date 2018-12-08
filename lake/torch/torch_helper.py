# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import json
import argparse
import lake.dir
import lake.file
from lake.option.optionParser import optionParser
import torch
import logging
import numpy as np
from recordtype import recordtype
import time
from . import network as torch_network
import numpy as np
from collections import defaultdict
import traceback


class TorchHelper(object):
	def __init__(self, outputs_path = './outputs/', output=None, option_dir='options', option_name=None, log_to_console=False, epoch_to_load=None):
		"""description
		Args:
			log_to_console: 显示到命令行，有其它模块设置了logging
		"""
		if option_dir is not None:
			sys.path.append(option_dir)
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
		parser.add_argument('--epoch_to_load', type=int, default=None, help='epoch_to_load')
		args, unknown = parser.parse_known_args()
		# args = parser.parse_args()
		return args, unknown

	def _load(self):
		args, unknown = self._parse_args()

		self._load_output_dir(args)
		self._load_opt(args, unknown)

		self.record_path = os.path.join(self._output_dir, 'record.txt')

		self._set_gpu()
		self._config_logging()
		self._load_epoch()

	# 设置self.output self._output_dir  self._outputs_path
	def _load_output_dir(self, args):
		# 确定输出目录，默认起一个时间
		# 命令行参数 > 传参 > 默认
		self._epoch_to_load = args.epoch_to_load
		if self._epoch_to_load:
			if len(args.output) < 0:
				raise Exception("There is no model to load!")
			else:
				if self._output is not None:
					self._epoch_to_load_path = self._output
				else:
					self._epoch_to_load_path = args.output
				self.output = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
				self._epoch_to_load_path = os.path.join(self._outputs_path, self._epoch_to_load_path)
		else:
			if len(args.output) > 0:
				self.output = args.output
			elif self._output is not None:
				self.output = self._output
			else:
				self.output = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

		self._output_dir = os.path.join(self._outputs_path, self.output)
		lake.dir.mk(self._outputs_path)
		lake.dir.mk(self._output_dir)

	# 设置self.opt
	def _load_opt(self, args, unknown):
		"""加载option，顺序为:
		1、使用命令行指定的
		2、使用保存目录里的
		3、默认: option_base
		后两者需要保存到_output_dir目录下
		"""
		self._option_path = os.path.join(self._output_dir, 'option.json')
		if len(args.option) > 0:
			option_name = args.option
		elif self._option_name is not None:
			option_name = self._option_name
		else:
			option_name = 'base'
		if os.path.exists(self._option_path):
			opt_network = None
		else:
			opt_pkg = __import__('option_' + option_name)
			opt_network = opt_pkg.Options()
		self.opt = optionParser(opt_network, self._option_path, args, unknown)

	def _set_gpu(self):
		if torch.cuda.is_available():
			torch_network.set_default_gpu_ids([int(item) for item in self.opt.gpu_ids])

	def _config_logging(self):
		self._logger = logging.getLogger(__name__)
		fh = logging.FileHandler(os.path.join(self._output_dir, 'train.log'))
		fh.setLevel(logging.INFO)
		formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s[line:%(lineno)d]: %(message)s')
		fh.setFormatter(formatter)
		self._logger.addHandler(fh)

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
			if self._epoch_to_load:
				model_path = os.path.join(self._epoch_to_load_path, '%d.pth' % self.opt.epoch_to_load)
				print("Load model from: " + model_path)
				if not os.path.isfile(model_path):
					raise ValueError('model %s not exits' % model_path)
			else:
				model_path = self.last_model_path()
			if model_path is not None:
				self._model.load_state_dict(torch.load(model_path))
				self._logger.info(u'model {} load successfully'.format(model_path))
			else:
				self._logger.info(u'model not load')
		except Exception as e:
			traceback.print_exc(file=sys.stdout)
			self._logger.info(u'model error')
			self.epoch = 1

	# 保存最好模型等操作
	@property
	def best_model_path(self):
		return os.path.join(self._output_dir, 'best.pth')

	def load_best_model(self):
		model_path = self.best_model_path
		if os.path.isfile(model_path):
			checkpoint = torch.load(model_path)
			self._model.load_state_dict(checkpoint['model'])
			return checkpoint['info']
		else:
			return None

	def _save_best_model(self, info):
		checkpoint = {}
		checkpoint['model'] = self._model.state_dict()
		checkpoint['info'] = info
		torch.save(checkpoint, self.best_model_path)

	def try_save_best_model(self, max_criteria, info):
		if not hasattr(self, 'max_criteria') or max_criteria >= self.max_criteria:
			self.max_criteria = max_criteria
			self._save_best_model(info)

	def _load_epoch(self):
		self.epoch = 1
		if self._epoch_to_load:
			self.epoch = self.opt.epoch_to_load + 1
			print("Afetr load model by epoch, start train from epoch: " + str(self.epoch))
			return
		if os.path.exists(self.record_path):
			records = lake.file.read(self.record_path)
			if len(records) > 0 and len(records[-1].strip()) > 0:
				self.epoch = int(json.loads(records[-1])['epoch']) + 1

	def default_optimizer(self, params=None, opt_type=None):
		opt = self.opt
		params = params or self._model.parameters()
		if opt_type == 'adam':
			optimizer = torch.optim.Adam(params, lr=opt.lr, weight_decay=opt.weight_decay)
		else:
			optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
		for group in optimizer.param_groups:
			group.setdefault('initial_lr', opt.lr)
		return optimizer

	def reset_lr(self, optimizer, lr):
		for group in optimizer.param_groups:
			group.setdefault('lr', lr)

	def _reset_record(self):
		self._epoch_records = defaultdict(list)
		self._epoch_start = time.time()

	def add_record(self, key, value):
		self._epoch_records[key].append(value)

	def add_records(self, records):
		for key, value in records.items():
			self.add_record(key, value)

	def mean_record(self, key):
		return np.mean(self._epoch_records[key])

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
		to_name = lambda x: os.path.basename(x).split('.')[0]
		paths = list(filter(lambda x: to_name(x).isdigit(), paths))
		if len(paths) > 0:
			epochs = np.array([int(to_name(x)) for x in paths])
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
		torch.save(self._model.state_dict(), self.new_model_path())
		self._logger.info('train finish')

	def finished(self):
		return self.epoch >= self.opt.epochs

# 已经验证此方法有效
def _init_weight(m):
	classname = m.__class__.__name__
	if classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
		# 权重初始化为均值为0标准差为0.02，即基本保持归一化后的分布
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)
	# xavier初始化
	elif classname.find('Conv') != -1:
		if hasattr(m, 'weight') and m.weight is not None:
			weight_shape = list(m.weight.data.size())
			fan_in = np.prod(weight_shape[1:4])
			fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
			w_bound = np.sqrt(6. / (fan_in + fan_out))
			m.weight.data.uniform_(-w_bound, w_bound)
		if hasattr(m, 'bias') and m.bias is not None:
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








