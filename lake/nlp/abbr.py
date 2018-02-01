# coding: utf-8
# 自然语言缩写名字的解释
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

# 相关参考
# 中文词性 http://repository.upenn.edu/cgi/viewcontent.cgi?article=1039&context=ircs_reports
# TODO: http://blog.csdn.net/eli00001/article/details/75088444
from tag_deps_data import _tag_names, _deps_names


_clear = lambda x: x.strip().lower()
tag_dict = dict([(_clear(k), v) for k, v in _tag_names])
deps_dict = dict([(_clear(k), v) for k, v in _deps_names])


def tag_description(tag):
	return tag_dict.get(_clear(tag), '未找到')

def deps_description(deps):
	return deps_dict.get(_clear(deps), '未找到')


if __name__ == '__main__':
	print(tag_description('VV'))
	print(tag_description('NN'))
	print(deps_description('root'))
	print(deps_description('conj'))