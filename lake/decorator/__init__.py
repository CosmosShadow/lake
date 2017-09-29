# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from .decorator_singleton import *
from .decorator_time import *
from .decorator_args import *


def register_fun(cls):
	"""注册函数func到类cls上"""
	def register_decorator(func):
		setattr(cls, func.func_name, func)
		return func
	return register_decorator