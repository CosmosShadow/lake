# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function
from .base_option import *


class NNOptions(Options):
	def __init__(self):
		super(NNOptions, self).__init__()
		self._default_parameters()

	def _default_parameters(self):
		self.parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
		self.parser.add_argument('--epochs', type=int, default=1e4, help='learn epochs')

		self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
		self.parser.add_argument('--clip_grad_norm', type=float, default=0.5, help='grad clip')
		self.parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight of nn decay')
		self.parser.add_argument('--momentum', type=float, default=0.9, help='weight of nn decay')

		self.parser.add_argument('--lr_decay', type=float, default=0.5, help='weight of nn decay')
		self.parser.add_argument('--lr_decay_start', type=int, default=0, help='weight of nn decay')
		self.parser.add_argument('--lr_decay_per', type=int, default=1e4, help='weight of nn decay')

		self.parser.add_argument('--option_name', type=str, default='', help='only placeholder')
		self.parser.add_argument('--output', type=str, default='', help='only placeholder')
		self.parser.add_argument('--epoch_to_load', type=int, default=None, help='epoch_to_load')
		self.parser.add_argument('--model_load_name', type=str, default='', help='')

		self.parser.add_argument('--test_per', type=int, default=100, help='test interval')
		self.parser.add_argument('--save_per', type=int, default=1e4, help='save interval')
		self.parser.add_argument('--print_per', type=int, default=100, help='print interval')
		# 根据当前GPU分配数据加载线程
		self.parser.add_argument('--num_workers', type=int, default=0, help='data loader num_workers')
		# 默认一块GPU，内部会检测cuda是否可用，仅可用时生效
		self.parser.add_argument('--gpu_ids', nargs='+', default=[0], help='gpu_ids, eg [0, 1]')

		self.parser.add_argument('--debug', type=int, default=0, help='')
