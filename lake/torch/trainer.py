# coding: utf-8
import os
import argparse
import lake
import torch
import logging


class Trainer(object):
	def __init__(self):
		self.data = None
		self.model = None
		self.optimizer = None
		self.hooks = []
		self._load()

	def _load(self):
		# 解析命令行输入
		parser = argparse.ArgumentParser()
		args = self.parser.parse_args()

		self._load_output_dir(args)
		self._load_opt(args)
		self._config_logging()

		self.save_path = self._output_dir + 'checkpoint.pth' % self.output
		self.csv_path = self._output_dir + 'log.csv' % self.output

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
			opt_pkg = __import__('option_' + name)
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

	def _check_train_components(self):
		"""检测训练要素"""
		assert self.data is not None
		assert self.model is not None
		try:
			self.model.load_network(self.save_path)
			print lake.string.color.red('load network success')
		except Exception as e:
			print lake.string.color.red('load newtork fail')
		self.optimizer = self.optimizer or torch.optim.Adam(self.model.parameters(), lr=self.opt.lr, weight_decay=self.opt.weight_decay)

	def add_hook(self, interval=1, fun=None):
		"""添加钩子: 训练过程中，间隔interval执行fun函数"""
		if fun is not None:
			self.hooks.append((interval, fun))

	def train(self):
		self._check_train_components()

		while self.epoch < self.epochs:
			batch = self.data.next()
			error = self.model.train(batch)
			self.optimizer.zero_grad()
			error.backward()
			for param in self.model.parameters():
				param.grad.data.clamp_(-self.clip_grad, self.clip_grad)
			self.optimizer.step()
			self.epoch += 1

			# TODO: 存储loss
			loss = error.data[0]

			# self._logger.info('start train')
			for interval, fun in self.hooks:
				if self.epoch % interval == 0:
					fun()

			if self.epoch % self.print_per == 0:
				# TODO: print_per
				pass

			if self.epoch % self.save_per == 0:
				self.model.save_network(self.save_path)

# 学习率衰减
# def exp_lr_scheduler(optimizer, epoch, lr_decay=0.1, lr_decay_epoch=7):
#     """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
#     if epoch % lr_decay_epoch:
#         return optimizer
    
#     for param_group in optimizer.param_groups:
#         param_group['lr'] *= lr_decay
#     return optimizer
