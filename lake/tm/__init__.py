# coding: utf-8
# 时间相关操作
from __future__ import absolute_import
from __future__ import print_function
import time

def datetime_str(the_time=None):
	the_time = the_time or time.localtime()
	return time.strftime("%Y-%m-%d %H-%M-%S", the_time)

def date_str(the_time=None):
	the_time = the_time or time.localtime()
	return time.strftime('%Y-%m-%d', the_time)

def date_str_simple(the_time=None):
	the_time = the_time or time.localtime()
	return time.strftime('%Y%m%d', the_time)

def date(the_time=None):
	pass


if __name__ == '__main__':
	print(date_str_simple())

	from datetime import datetime
	print(datetime.now().month)
	print(datetime(2018, 1, 2))
	print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
	print(type(datetime.now()))
	print(datetime.now().date())
