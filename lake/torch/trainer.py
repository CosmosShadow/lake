# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import json
import argparse
import lake.dir
import lake.file
from lake.tm import Timer
import torch
import logging
import numpy as np
from recordtype import recordtype
import time
from . import network as torch_network


class Trainer(object):
	def __init__(self, output=None, option_name=None, log_to_console=False, epoch_to_load=None):
		"""description
		Args:
			log_to_console: 显示到命令行，有其它模块设置了logging
		"""
		self.log_to_console = log_to_console
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

		self.record_path = self._output_dir + 'record.txt'
		self.save_path = self._output_dir + 'checkpoint.pth'

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
			self.opt = recordtype('X', option_dict.keys())(*option_dict.values())
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
			if self._epoch_to_load is not None:
				model_path = os.path.join(self._output_dir, '%d.pth' % self._epoch_to_load)
				if not os.path.isfile(model_path):
					raise ValueError('你想加载的模型%s不存在' % model_path)
			else:
				model_path = self.last_model_path()
			if model_path is not None:
				print(model_path)
				self._model.load_network(model_path)
				self._logger.info('模型加载成功')
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
		for key, value in records.items():
			self.add_record(key, value)

	def _epoch_log(self, values):
		results = ['epoch: %d' % self.epoch]
		for key, value in values.items():
			if key != 'epoch':
				value = ('%.6f' % value) if isinstance(value, float) else str(value)
				results.append('%s: %s' % (key, value))
		if self.has_test:
			self._logger.info('-' * 10)
		self._logger.info('   '.join(results))

	def _store_record(self):
		self.add_record('epoch', self.epoch)
		if hasattr(self, 'current_lr'):
			self.add_record('lr', self.current_lr)
		self.add_record('time', time.time() - self._epoch_start)
		if self.epoch % self.opt.print_per == 0:
			self._epoch_log(self._epoch_records)
		record_json = json.dumps(self._epoch_records)
		lake.file.add_line(record_json, self.record_path)
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
			for _ in range(self.data_test.count()):
				batch = self.data_test.next()
				result = self._model.test_step(self.epoch, batch)
				results.append(result)
			results_average = {}
			for key in results[0].keys():
				results_average['test_' + key] = np.mean([item[key] for item in results[0]])
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
		_t = Timer()
		loop_per_epoch = self.data_train.count()
		for index in range(loop_per_epoch):
			batch = self.data_train.next() if self.data_train is not None else None
			_t.tic()
			train_dict = self._model.train_step(self.epoch, batch)
			time_FP = _t.toc()

			if self.optimizer is None:
				self.add_records(train_dict)
			else:
				error = train_dict['loss']
				self.optimizer.zero_grad()
				_t.tic()
				error.backward()
				time_BP = _t.toc()
				if self.opt.clip_grad_norm > 0:
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt.clip_grad_norm)
				self.optimizer.step()

				self.add_record('loss', float(error.data.cpu().item()))
				for key, value in train_dict.items():
					if key != 'loss':
						self.add_record(key, value)
			records = {
				'loss': '%07.4f' % (train_dict['loss'].item()),
				'rpn_class_loss': '%07.4f' % (train_dict['rpn_class_loss']),
				'rpn_bbox_loss': '%07.4f' % (train_dict['rpn_bbox_loss']),
				'mrcnn_class_loss': '%07.4f' % (train_dict['mrcnn_class_loss']),
				'mrcnn_bbox_loss': '%07.4f' % (train_dict['mrcnn_bbox_loss']),
				'mrcnn_mask_loss': '%07.4f' % (train_dict['mrcnn_mask_loss']),
				'time_FP': '%04.2f' % (time_FP),
				'time_BP': '%04.2f' % (time_BP)
			}
			if self.opt.debug != 1:
				print('epoch:{:<3}{:>4}/{:<4}: {}'.format(self.epoch, index + 1, loop_per_epoch, records))

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

	def train(self):
		self._check_train_components()
		self._update_lr(force=True)
		self._logger.info('train start')
		self._model.train_start()

		while self.epoch <= self.opt.epochs:
			self.has_test = self.epoch >= self.opt.test_per and self.epoch % self.opt.test_per == 0
			if self.has_test:
				# self._model.eval()
				self._test()

			# self._model.train()
			self._train()
			if self.epoch % self.opt.save_per == 0:
				self._model.save_network(self.new_model_path())
				self.add_record('save', 1)

			self._run_hook()
			self._store_record()
			self._update_lr()
			self.epoch += 1

		self._model.save_network(self.new_model_path())
		self._model.train_finish()
		self._logger.info('train finish')

