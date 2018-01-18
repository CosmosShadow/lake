# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function
from .file_add import *
from .file_op import *
from .file_read import *
from .file_write import *
from .file_pkg import *


# 序列化
def serialize(data, path):
	import pickle
	with open(path, 'wb') as f:
		pickle.dump(data, f)

# 反序列化
def unserialize(path):
	import pickle
	with open(path, 'rb') as f:
		return pickle.load(f)
