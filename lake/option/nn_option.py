# coding: utf-8
from base_option import *


class NNOptions(Options):
	def __init__(self):
		super(NNOptions, self).__init__()
		self._default_parameters()

	def _default_parameters(self):
		self.parser.add_argument('--lr', type=float, default=1e-2, help='学习率')
		self.parser.add_argument('--clip_grad', type=float, default=0.1, help='学习率')
		self.parser.add_argument('--weight_decay', type=float, default=1e-5, help='学习率')
		self.parser.add_argument('--epochs', type=int, default=1e4, help='学习轮数')

		self.parser.add_argument('--output', type=str, default='tmp', help='输出地址')
		self.parser.add_argument('--save_per', type=int, default=100, help='保存间隔')
		self.parser.add_argument('--print_per', type=int, default=10, help='输出间隔')