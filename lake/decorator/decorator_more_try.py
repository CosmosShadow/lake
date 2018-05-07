# coding: utf-8
# 多次尝试
import time

def more_try(times, sleep_time):
	"""多次尝试装饰器
	Args:
		times: 尝试次数
		sleep_time: 每次睡眠时间
	"""
	def decorator(func):
		def wrapper(*args, **kwargs):
			retry = 0
			exception = None
			while retry < times:
				try:
					return func(*args, **kwargs)
				except Exception as e:
					exception = e
					time.sleep(sleep_time)
					retry += 1
			else:
				raise exception
		return wrapper
	return decorator