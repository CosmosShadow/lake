# coding: utf-8
import argparse


class Options(object):
	def __init__(self):
		self.parser = argparse.ArgumentParser(conflict_handler='resolve')
		self.initialized = False

	def initialize(self):
		if __name__ == '__main__':
			self.parser.add_argument('--int', type=int, default=1, help='整数')
			self.parser.add_argument('--float', type=float, default=0.5, help='浮点数')
			self.parser.add_argument('--str', type=str, default='str', help='字符串')
			self.parser.add_argument('--boolean', action='store_true', default=True, help='布尔值')
			self.parser.add_argument('--list', nargs='+', default=['hello', 256, 8], help='数组')
			# 强制需求: required=True

	def show(self):
		print('------------ Options -------------')
		for k, v in sorted(vars(self.opt).items()):
			print("{:30} {}".format(k, v))
		print('-------------- End ----------------')

	def __call__(self):
		if not self.initialized:
			self.initialize()
			self.initialized = True
		self.opt = self.parser.parse_args()
		self.show()
		return self.opt


if __name__ == '__main__':
	opt = Options()()

