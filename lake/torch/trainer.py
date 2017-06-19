# coding: utf-8
import lake
import torch


class Trainer(object):
	def __init__(self, data, model, optimizer=None, opt=None):
		self.opt = opt
		self.save_per = getattr(opt, 'save_per', 100)
		self.print_per = getattr(opt, 'print_per', 10)
		self.clip_grad = getattr(opt, 'clip_grad', 0.1)
		lr = getattr(opt, 'lr', 1e-3)
		weight_decay = getattr(opt, 'weight_decay', 1e-5)

		self.name = getattr(opt, 'name', 'tmp')
		self.epochs = getattr(opt, 'epochs', 10000)

		self.data = data
		self.model = model
		self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

		self.hooks = []
		self._load_env()

	def _load_env(self):
		# TODO: current epoch
		# TODO: logger
		self.save_path = './outputs/%s/checkpoint.pth' % self.name
		lake.dir.check_dir('./outputs/')
		lake.dir.check_dir(self.save_path)

		try:
			self.model.load_network(self.save_path)
			print lake.string.color.red('load network success')
		except Exception as e:
			print lake.string.color.red('load newtork fail')

	def hook(self, interval=1, fun=None):
		if fun is not None:
			self.hooks.append((interval, fun))

	def run(self):
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
