# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function
from .base_option import *


class NNOptions(Options):
	def __init__(self):
		super(NNOptions, self).__init__()
		self._default_parameters()

	def _default_parameters(self):
		self.parser.add_argument('--batch_size', type=int, default=2, help='batch_size')
		self.parser.add_argument('--epochs', type=int, default=1e4, help='learn epochs')

		self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
		self.parser.add_argument('--clip_grad_norm', type=float, default=0.5, help='grad clip')
		self.parser.add_argument('--weight_decay', type=float, default=1e-7, help='weight of nn decay')

		self.parser.add_argument('--lr_decay', type=float, default=0.5, help='weight of nn decay')
		self.parser.add_argument('--lr_decay_start', type=int, default=0, help='weight of nn decay')
		self.parser.add_argument('--lr_decay_per', type=int, default=1e4, help='weight of nn decay')

		self.parser.add_argument('--option', type=str, default='base', help='only placeholder')
		self.parser.add_argument('--output', type=str, default='', help='only placeholder')

		self.parser.add_argument('--save_per', type=int, default=1e4, help='save interval')
		self.parser.add_argument('--print_per', type=int, default=100, help='print interval')

		# 默认一块GPU，内部会检测cuda是否可用，仅可用时生效
		self.parser.add_argument('--gpu_ids', nargs='+', default=[0], help='gpu_ids, eg [0, 1]')
