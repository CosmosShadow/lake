# coding: utf-8
import sys

class BaseLearningRator(object):
	def __init__(self, lr, epoch=0):
		self.start_lr = lr
		self.epoch = epoch
		self.optimizers = []
		self.current_lr = self.start_lr

	def apply(self, optimizer):
		optimizers = optimizer if isinstance(optimizer, list) else [optimizer]
		for opti in optimizers:
			if opti not in self.optimizers:
				self.optimizers.append(opti)
		self.update_lr()

	def step(self):
		self.epoch += 1
		self._step()
		
	def update_lr(self):
		lr = self.lr()
		for opti in self.optimizers:
			for param_group in opti.param_groups:
				param_group['lr'] = lr

	def _step(self):
		pass

	def lr(self):
		return self.current_lr


class DecayLearningRator(BaseLearningRator):
	def __init__(self, lr, epoch=0, decay_start=0, decay_step=1, decay_rate=1.0):
		super(DecayLearningRator, self).__init__(lr, epoch)
		self.decay_start = decay_start
		self.decay_step = decay_step
		self.decay_rate = decay_rate
		self._step()

	def _step(self):
		update = False
		
		if self.epoch < self.decay_start:
			self.current_lr = self.start_lr
		else:
			if (self.epoch - self.decay_start) % self.decay_step == 0:
				update = True
			self.current_lr = self.start_lr * (self.decay_rate ** ((self.epoch - self.decay_start) / self.decay_step + 1))

		if update == True:
			self.update_lr()





























