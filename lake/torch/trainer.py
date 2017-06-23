# coding: utf-8
import os
import argparse
import lake
import torch
import logging


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
		args = self.parser.parse_args()

		self._load_output_dir(args)
		self.save_path = self._output_dir + 'checkpoint.pth' % self.output
		self.record_path = self._output_dir + 'record.txt' % self.output

		self._load_opt(args)
		self._config_logging()
		self._load_epoch()

	def _load_output_dir(self, args):
		# 确定输出目录，默认 tmp
		if hasattr(args, 'output'):
			self.output = args.output
		else:
			self.output = 'tmp'
		self._output_dir  = './outputs/%s/' + self.output

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
			opt_pkg = __import__('options.option_' + name)
			self.opt = opt_pkg.Options()()
			option_json = json.dumps(vars(self.opt), indent=4)
			lake.file.write(option_json, self._option_path)

	def _config_logging(self):
		log_path = self._output_dir + 'train.log'
		logging.basicConfig(
				filename = log_path,
				filemode = 'a',
				level = logging.DEBUG,
				format = '%(asctime)s - %(levelname)s - %(name)s[line:%(lineno)d]: %(message)s')
		self._logger = logging.getLogger()

	def _load_epoch(self):
		self.epoch = 1
		try:
			records = lake.file.read(self.record_path)
			self.epoch = int(json.load(records[-1])['epoch'])
			self._logger.info('load epoch from record.txt')
		except Exception as e:
			self._logger.info(e)
		self._logger.info('start epoch %d' % self.epoch)


	def _check_train_components(self):
		"""检测训练要素"""
		assert self.data_train is not None
		assert self.model is not None
		try:
			self.model.load_network(self.save_path)
			self._logger.info('load network success')
		except Exception as e:
			self._logger.info('load network fail')
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

	def add_record(self, key, value):
		self._epoch_records[key] = value

	def add_records(self, records):
		for key, value in records.iteritems():
			self.add_record(key, value)

	def _store_record(self):
		self._epoch_records['epoch'] = self.epoch
		record_json = json.dumps(self._epoch_records)
		lake.file.add_txt(record_json, self.record_path)
		self._clear_record()
		self._logger.info(record_json)

	def train(self):
		self._check_train_components()

		self._logger.info('train start')

		train_batch_count = self.data_train.count(self.opt.batch_size)

		while self.epoch <= self.epochs:
			batch = self.data_train.next(self.opt.batch_size)
			train_dict = self.model.train(batch)
			error = train_dict['error']
			self.optimizer.zero_grad()
			error.backward()
			for param in self.model.parameters():
				param.grad.data.clamp_(-self.clip_grad, self.clip_grad)
			self.optimizer.step()

			loss = error.data[0]
			self.add_record('loss', loss)
			for key, value in train_dict.iteritems():
				if key ~= 'error':
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
