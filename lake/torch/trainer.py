# coding: utf-8
import lake
import torch


class Trainer(object):
	def __init__(self, name, data, model, optimizer=None, opt=None):
		self.opt = opt
		self.save_per = getattr(opt, 'save_per', 100)
		self.print_per = getattr(opt, 'print_per', 10)
		self.clip_grad = getattr(opt, 'clip_grad', 0.01)
		lr = getattr(opt, 'lr', 1e-3)
		weight_decay = getattr(opt, 'weight_decay', 1e-5)

		self.name = name
		self.data = data
		self.model = model
		self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

		self.hooks = []
		self._load_env()

	def _load_env(self):
		# current epoch
		# logger
		self.save_path = './outputs/%s/checkpoint.pth' % self.name
		lake.dir.check_dir('./outputs/')
		lake.dir.check_dir(self.save_path)

		try:
			self.model.load_network(self.save_path)
			print lake.string.color.red('load network success')
		except Exception as e:
			print lake.string.color.red('load newtork fail')

	def hook(self, fun, interval=1):
		self.hooks.append((interval, fun))

	def run(self, epochs):
		while self.epoch < epochs:	
			batch = self.data.next()
			error = self.model.train(batch)
			self.optimizer.zero_grad()
			error.backward()
			for param in self.model.parameters():
				param.grad.data.clamp_(-self.clip_grad, self.clip_grad)
			self.optimizer.step()
			self.epoch += 1

			for interval, fun in self.hooks:
				if self.epoch % interval == 0:
					fun()

			if self.epoch % self.print_per == 0:
				# print_per
				pass

			if self.epoch % self.save_per == 0:
				self.model.save_network(self.save_path)

