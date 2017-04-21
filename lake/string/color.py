# coding: utf-8

# 黑色字
def black(str):
	return '\033[0;30m%s\033[0m' % str

# 红色字
def red(str):
	return '\033[0;31m%s\033[0m' % str

# 绿色字
def green(str):
	return '\033[0;32m%s\033[0m' % str

# 黄色字
def yellow(str):
	return '\033[0;33m%s\033[0m' % str

# 蓝色字
def blue(str):
	return '\033[0;34m%s\033[0m' % str

# 紫色字
def purple(str):
	return '\033[0;35m%s\033[0m' % str

# 天蓝字
def blue_sky(str):
	return '\033[0;36m%s\033[0m' % str

# 白色字
def white(str):
	return '\033[0;37m%s\033[0m' % str