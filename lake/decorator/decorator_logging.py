# coding: utf-8
# 使用装饰器来对函数进行自动logging

def logging_fun(log_fun, comment=''):
	"""
	Args:
		a : log_fun, eg. logging.error
		comment: 注释
	Waning: 函数参数需能够进行json化
	"""
	import json
	
	def _deco(func):
		def __deco(*args, **kwargs):
			message = {
				'函数名': func.func_name,
				'注释': comment,
				'列表参数': json.dumps(args, ensure_ascii=False),
				'字典参数': json.dumps(kwargs, ensure_ascii=False)
			}
			log_fun(json.dumps(message, ensure_ascii=False))
			return func(*args, **kwargs)
		return __deco  
	return _deco


if __name__ == '__main__':
	import logging

	@logging_fun(logging.error, 'comment')
	def hello(arg):
		print(arg)

	hello('world')