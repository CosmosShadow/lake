# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function
from .file_add import *
from .file_op import *
from .file_read import *
from .file_write import *
from .file_pkg import *


def exists(path):
	# 是否存在
	import os
	return os.path.exists(path)

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

# 文件md5
def md5(file_path):
	import hashlib
	import os
	f = open(file_path,'rb')  
	md5_obj = hashlib.md5()
	while True:
		d = f.read(8096)
		if not d:
			break
		md5_obj.update(d)
	hash_code = md5_obj.hexdigest()
	f.close()
	md5_str = str(hash_code).lower()
	return md5_str







