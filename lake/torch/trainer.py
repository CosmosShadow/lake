# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import json
import argparse
import lake
import torch
import logging
import numpy as np
from collections import namedtuple
import time
from . import network as torch_network


class Trainer(object):
	def __init__(self, log_to_console=True):
		"""description
		Args:
			log_to_console: 显示到命令行，有其它模块设置了logging
		"""
		self.log_to_console = log_to_console
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

		self.record_path = self._output_dir + 'record.txt'
		self.save_path = self._output_dir + 'checkpoint.pth'

		self._set_gpu()
		self._config_logging()
		self._load_epoch()

	def _load_output_dir(self, args):
		# 确定输出目录，默认起一个时间
		if len(args.output) > 0:
			self.output = args.output
		else:
			self.output = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

		self._output_dir  = './outputs/%s/' % self.output
		lake.dir.mk('./outputs/')
		lake.dir.mk(self._output_dir)

	def _load_opt(self, args):
		"""加载option，顺序为:
		1、使用保存目录里的
		2、使用命令行指定的
		3、默认: option_base
		后两者需要保存到_output_dir目录下
		"""
		self._option_path = self._output_dir + 'option.json'
		if os.path.exists(self._option_path):
			option_json = lake.file.read(self._option_path)
			option_dict = json.loads(option_json)
			self.opt = namedtuple('X', option_dict.keys())(*option_dict.values())
		else:
			option_name = args.option if len(args.option) > 0 else 'base'
			sys.path.append('options')
			opt_pkg = __import__('option_' + option_name)
			self.opt = opt_pkg.Options()()
			option_json = json.dumps(vars(self.opt), indent=4)
			lake.file.write(option_json, self._option_path)

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

	@model.setter
	def model(self, value):
		self._model = value
		self._model.output_dir = self._output_dir
		try:
			self._model.load_network(self.save_path)
			self._logger.info('load network success')
		except Exception as e:
			print(e)
			self._logger.info('load network fail')
			self.epoch = 1

	def _load_epoch(self):
		self.epoch = 1
		if os.path.exists(self.record_path):
			records = lake.file.read(self.record_path)
			if len(records) > 0 and len(records[-1].strip()) > 0:
				self.epoch = int(json.loads(records[-1])['epoch'])

	def _check_train_components(self):
		"""检测训练要素"""
		assert self._model is not None
		self._logger.info(self._model)
		if self.optimizer is None and not self._model.use_inner_optimizer:
			self.optimizer = torch.optim.Adam(
					self._model.model.parameters(),
					lr=self.opt.lr,
					weight_decay=self.opt.weight_decay)

	def add_hook(self, interval=1, fun=None):
		"""添加钩子: 训练过程中，间隔interval执行fun函数"""
		if fun is not None:
			self.hooks.append((interval, fun))

	def _run_hook(self):
		for interval, fun in self.hooks:
			if self.epoch % interval == 0:
				fun()

	def _reset_record(self):
		self._epoch_records = {}
		self._epoch_start = time.time()

	def add_record(self, key, value):
		if isinstance(value, float):
			value = round(value, 6)
		self._epoch_records[key] = value

	def add_records(self, records):
		for key, value in records.iteritems():
			self.add_record(key, value)

	def _epoch_log(self, values):
		results = ['epoch: %d' % self.epoch]
		for key, value in values.iteritems():
			if key != 'epoch':
				value = ('%.6f' % value) if isinstance(value, float) else str(value)
				results.append('%s: %s' % (key, value))
		self._logger.info('   '.join(results))

	def _store_record(self):
		self.add_record('epoch', self.epoch)
		if hasattr(self, 'current_lr'):
			self.add_record('lr', self.current_lr)
		self.add_record('time', time.time() - self._epoch_start)
		record_json = json.dumps(self._epoch_records)
		lake.file.add_line(record_json, self.record_path)
		if self.epoch % self.opt.print_per == 0:
			self._epoch_log(self._epoch_records)
		self._reset_record()

	def _update_lr(self, force=False):
		if self.optimizer is not None:
			step = self.epoch - self.opt.lr_decay_start
			if force or (step > 0 and step % self.opt.lr_decay_per == 0):
				self.current_lr = self.opt.lr * (self.opt.lr_decay ** (step / self.opt.lr_decay_per))
				for param_group in self.optimizer.param_groups:
					param_group['lr'] = self.current_lr

	def _test(self):
		if self.data_test is not None:
			results = []
			for _ in range(self.data_test.count(self.opt.batch_size)):
				batch = self.data_test.next(self.opt.batch_size)
				result = self._model.test_step(self.epoch, batch)
				results.append(result)
			results_average = {}
			for key in results[0].keys():
				results_average['test_' + key] = np.mean([item[key] for item in results])
			self.add_records(results_average)
			if self.epoch % self.opt.print_per != 0:
				self._epoch_log(results_average)
		else:
			result = self._model.test_step(self.epoch, None)
			if result is not None:
				result = dict(zip(['test_' + key for key in result.keys()], result.values()))
				self.add_records(result)
				if self.epoch % self.opt.print_per != 0:
					self._epoch_log(result)


	def _train(self):
		batch = self.data_train.next(self.opt.batch_size) if self.data_train is not None else None
		train_dict = self._model.train_step(self.epoch, batch)

		if self.optimizer is None:
			self.add_records(train_dict)
		else:
			error = train_dict['loss']
			self.optimizer.zero_grad()
			error.backward()
			if self.opt.clip_grad_norm > 0:
				torch.nn.utils.clip_grad_norm(self.model.parameters(), self.opt.clip_grad_norm)
			self.optimizer.step()

			self.add_record('loss', float(error.data[0]))
			for key, value in train_dict.iteritems():
				if key != 'loss':
					self.add_record(key, value)


	def train(self):
		self._check_train_components()
		self._update_lr(force=True)
		self._logger.info('train start')
		self._model.train_start()

		while self.epoch <= self.opt.epochs:
			if self.data_train is not None:
				train_batch_count = self.data_train.count(self.opt.batch_size)
				if self.epoch % train_batch_count == 0:
					self._model.eval()
					self._test()

			self._model.train()
			self._train()

			if self.epoch % self.opt.save_per == 0:
				self._model.save_network(self.save_path)
				self.add_record('save', 1)

			self._run_hook()
			self._store_record()
			self._update_lr()
			self.epoch += 1

		self._model.save_network(self.save_path)
		self._model.train_finish()
		self._logger.info('train finish')

