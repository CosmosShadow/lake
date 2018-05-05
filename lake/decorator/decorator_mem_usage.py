# coding: utf-8
# 内存使用情况装饰器
import logging
_logger = logging.getLogger(__name__)

# 如果logging没有设置，默认是ERROR。
if len(logging.getLogger().handlers) == 0:
	logging.basicConfig(level=logging.DEBUG)


def get_current_mem_usage():
	import psutil
	import os
	process = psutil.Process(os.getpid())
	return process.memory_info().rss


def size_2_human(mem_usage):
	return '%.3f M' % (mem_usage / 1024.0 / 1024.0)


def log_mem_usage(prefix=''):
	now_size = size_2_human(get_current_mem_usage())
	_logger.info(' %s当前内存使用: %s' % (prefix, now_size))


def mem_usage(func):
	def _fun(*args, **kwargs):
		past_mem = get_current_mem_usage()
		ret = func(*args, **kwargs)
		now_mem = get_current_mem_usage()

		past_size = size_2_human(past_mem)
		add_size = size_2_human(now_mem - past_mem)
		new_size = size_2_human(now_mem)
		_logger.info(' 内存状态    函数名: %s    原本: %s    新增: %s    最新: %s' % (func.func_name, past_size, add_size, new_size))

		return ret
	return _fun


if __name__ == '__main__':
	@mem_usage
	def test():
		a = [0] * 100000
		return a

	test()

	# 输出: 
	# 内存状态    函数名: test    原本: 7.281 M    新增: 0.766 M    最新: 8.047 M

	# 如下进行import lake时，输出: 
	# 内存状态    函数名: test    原本: 55.176 M    新增: 0.766 M    最新: 55.941 M
	# 说明lake import 了太多不必要的库，造成内存大涨。
	# 特别是在多进程(如10线程)服务端时，带来 50 * 10 = 500 M 的开销

# # coding: utf-8
# import lake

# @lake.decorator.mem_usage
# def test():
# 	a = [0] * 100000
# 	return a

# test()

	# 注释掉 lake.__init__ 下所以import后，输出: 
	# 内存状态    函数名: test    原本: 7.355 M    新增: 0.766 M    最新: 8.121 M
	
# # coding: utf-8
# from lake import decorator

# @decorator.mem_usage
# def test():
# 	a = [0] * 100000
# 	return a

# test()


