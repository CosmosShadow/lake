# coding: utf-8
# 时间相关操作
from __future__ import absolute_import
from __future__ import print_function
import datetime

datetime_format = '%Y-%m-%d %H:%M:%S'
date_format = '%Y-%m-%d'
simple_date_format = '%Y%m%d'


def datetime_str(the_time=None):
	# 年-月-日 时:分:秒
	the_time = the_time or datetime.datetime.now()
	return the_time.strftime(datetime_format)


def date_str(the_date=None):
	# 年-月-日
	the_date = the_date or datetime.date.today()
	return the_date.strftime(date_format)


def date_str_simple(the_date=None):
	# 年月日
	the_date = the_date or datetime.date.today()
	return the_date.strftime(simple_date_format)


def str_2_datetime(date_str):
	return datetime.datetime.strptime(date_str, datetime_format)


if __name__ == '__main__':
	print(datetime_str())
	print(date_str())
	print(date_str_simple())

	print(datetime.datetime.now().month)
	print(datetime.datetime(2018, 1, 2))
	print(datetime.datetime.now().strftime(datetime_format))
	print(type(datetime.datetime.now()))
	print(datetime.datetime.now().date())
	print(datetime.datetime.now().date().strftime(datetime_format))
	print(datetime.datetime.now().date().timetuple())
	print(datetime.datetime.now() + datetime.timedelta(days=1))
	print(datetime.datetime.now().date() + datetime.timedelta(days=1))
	print(type(datetime.date.today()))

	print (str_2_datetime('2018-05-02 15:07:00'))
