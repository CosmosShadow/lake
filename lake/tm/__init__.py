# coding: utf-8
# 时间相关操作
from __future__ import absolute_import
from __future__ import print_function
import time

def current_datetime_str():
	return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

def current_date_str():
	return time.strftime('%Y-%m-%d', time.localtime())