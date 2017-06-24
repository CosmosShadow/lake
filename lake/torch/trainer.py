# coding: utf-8
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


class Trainer(object):
	def __init__(self):
		self.data_train = None
		self.data_test = None
		self.model = None
		self.optimizer = None
		self.hooks = []
		self._load()
		self._clear_record()

	def _load(self):
		# 解析命令行输入
		parser = argparse.ArgumentParser()
		args = parser.parse_args()

		self._load_output_dir(args)
		self.save_path = self._output_dir + 'checkpoint.pth'
		self.record_path = self._output_dir + 'record.txt'

		self._load_opt(args)
		self._config_logging()
		self._load_epoch()

	def _load_output_dir(self, args):
		# 确定输出目录，默认 tmp
		if hasattr(args, 'output'):
			self.output = args.output
		else:
			self.output = 'tmp'
		self._output_dir  = './outputs/%s/' % self.output

		lake.dir.check_dir('./outputs/')
		lake.dir.check_dir(self._output_dir)

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
			option_name = args.option if hasattr(args, 'option') else 'base'
			sys.path.append('options')
			opt_pkg = __import__('option_' + option_name)
			self.opt = opt_pkg.Options()()
			option_json = json.dumps(vars(self.opt), indent=4)
			lake.file.write(option_json, self._option_path)

	def _config_logging(self):
		log_path = self._output_dir + 'train.log'
		format = '%(asctime)s - %(levelname)s - %(name)s[line:%(lineno)d]: %(message)s'

		# 文件记录
		logging.basicConfig(
				filename = log_path,
				stream = sys.stdout,
				filemode = 'a',
				level = logging.DEBUG,
				format = format)

		# 控制台输出
		root = logging.getLogger()
		ch = logging.StreamHandler(sys.stdout)
		ch.setLevel(logging.DEBUG)
		formatter = logging.Formatter(format)
		ch.setFormatter(formatter)
		root.addHandler(ch)

		self._logger = logging.getLogger(__name__)

	def _load_epoch(self):
		self.epoch = 1
		if os.path.exists(self.record_path):
			records = lake.file.read(self.record_path)
			if len(records) > 0 and len(records[-1].strip()) > 0:
				self.epoch = int(json.loads(records[-1])['epoch'])

	def _check_train_components(self):
		"""检测训练要素"""
		assert self.data_train is not None
		assert self.model is not None
		try:
			self.model.load_network(self.save_path)
			self._logger.info('load network success')
		except Exception as e:
			self._logger.info('load network fail')
			self.epoch = 1
		self.optimizer = self.optimizer or torch.optim.Adam(self.model.parameters(), lr=self.opt.lr, weight_decay=self.opt.weight_decay)

	def add_hook(self, interval=1, fun=None):
		"""添加钩子: 训练过程中，间隔interval执行fun函数"""
		if fun is not None:
			self.hooks.append((interval, fun))

	def _run_hook(self):
		for interval, fun in self.hooks:
			if self.epoch % interval == 0:
				fun()

	def _clear_record(self):
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
				value = str(round(value, 6)) if isinstance(value, float) else str(value)
				results.append('%s: %s' % (key, value))
		self._logger.info('   '.join(results))

	def _store_record(self):
		self.add_record('epoch', self.epoch)
		self.add_record('time', time.time() - self._epoch_start)
		record_json = json.dumps(self._epoch_records)
		lake.file.add_line(record_json, self.record_path)
		if self.epoch % self.opt.print_per == 0:
			self._epoch_log(self._epoch_records)
		self._clear_record()
		# 每100才记录一次
		# if self.epoch % self.opt.print_per == 0:
		# 	self.add_record('epoch', self.epoch)
		# 	self.add_record('time', time.time() - self._epoch_start)
		# 	record_json = json.dumps(self._epoch_records)
		# 	lake.file.add_line(record_json, self.record_path)
		# 	self._epoch_log(self._epoch_records)
		# 	self._clear_record()

	def train(self):
		self._check_train_components()

		self._logger.info('train start')

		train_batch_count = self.data_train.count(self.opt.batch_size)

		while self.epoch <= self.opt.epochs:
			batch = self.data_train.next(self.opt.batch_size)
			train_dict = self.model.train(batch)
			error = train_dict['loss']
			self.optimizer.zero_grad()
			error.backward()
			for param in self.model.parameters():
				param.grad.data.clamp_(-self.opt.clip_grad, self.opt.clip_grad)
			self.optimizer.step()

			self.add_record('loss', float(error.data[0]))
			for key, value in train_dict.iteritems():
				if key != 'loss':
					self.add_record(key, value)

			if self.epoch % self.opt.save_per == 0:
				self.model.save_network(self.save_path)
				self.add_record('save', 1)

			if self.epoch % train_batch_count == 0 and self.data_test is not None:
				results = []
				for _ in range(self.data_test.count(self.opt.batch_size)):
					batch = self.data_test.next(self.opt.batch_size)
					result = self.model.test(batch)
					results.append(result)
				results_average = {}
				for key in results[0].keys():
					results_average['test_' + key] = np.mean([item[key] for item in results])
				self.add_records(results_average)
				self._epoch_log(results_average)

			self._run_hook()
			self._store_record()

			self.epoch += 1

		self._logger.info('train finish')


# 学习率衰减
# def exp_lr_scheduler(optimizer, epoch, lr_decay=0.1, lr_decay_epoch=7):
#     """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
#     if epoch % lr_decay_epoch:
#         return optimizer
    
#     for param_group in optimizer.param_groups:
#         param_group['lr'] *= lr_decay
#     return optimizer
