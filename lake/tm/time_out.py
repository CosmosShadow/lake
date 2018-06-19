# coding: utf-8
# created by lichen(lichenarthurdata@gmail.com)
# 运行超时器
from __future__ import absolute_import
from __future__ import print_function
import signal


class TimeOutError(Exception):
	pass


class TimeOut(object):
	def __init__(self, time_in_seconds):
		super(TimeOut, self).__init__()
		self._time_in_seconds = time_in_seconds

	def __enter__(self):
		# 收到信号 SIGALRM 后的回调函数，第一个参数是信号的数字，第二个参数是the interrupted stack frame.
		def handle(signum, frame):
			raise TimeOutError('运行超时')
		  # 设置信号和回调函数
		signal.signal(signal.SIGALRM, handle)
		  # 设置time_in_seconds秒的闹钟
		signal.alarm(self._time_in_seconds)

	def __exit__(self, type, value, trace):
		signal.alarm(0)


if __name__ == '__main__':
	import time
	try:
		with TimeOut(1) as t:
			time.sleep(2)
			print('ok')
	except Exception as e:
		print(isinstance(e, TimeOutError))
		print(e.message)
		print('超时')