# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from .decorator_singleton import *
from .decorator_time import *
from .decorator_args import *
from .decorator_logging import *
from .decorator_mem_usage import mem_usage


def register_fun(cls):
	"""注册函数func到类cls上"""
	def register_decorator(func):
		setattr(cls, func.func_name, func)
		return func
	return register_decorator


def empty_decorator(func):
	# 空装饰器
	def _fun(*args, **kwargs):
		return func(*args, **kwargs)
	return _fun