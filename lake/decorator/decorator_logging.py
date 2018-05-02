# coding: utf-8
# 使用装饰器来对函数进行自动logging
import json

def logging_fun(log_fun, comment=None):  
	def _deco(func):  
		def __deco(*args, **kwargs):
			message = '\n函数名: ' + func.func_name + '\n参数: ' + json.dumps(args) + ' | ' + json.dumps(kwargs) + '\n'
			if comment is not None:
				message = '    注释: ' + comment + message
			log_fun(message)
			return func(*args, **kwargs)
		return __deco  
	return _deco


if __name__ == '__main__':
	import logging

	@logging_fun(logging.error, 'comment')
	def hello(arg):
		print arg

	hello('world')