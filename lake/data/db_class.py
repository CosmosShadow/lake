# coding: utf-8
# 数据库类
# 一个管理类与数据库字段映射的小工具
# 有时间看看这个库: https://github.com/coleifer/peewee
import json
import copy

class DBClass(object):
	# fields = 'id, contents, status'
	# # 每一个都应有loads与dumps函数
	# field_parses = {
	# 	'contents': json,
	# }
	# default_values = {
	# 	'status': 1
	# }

	fields = ''
	field_parses = {}
	default_values = {}

	def __init__(self, item=None, mapping=None):
		# item为可遍历的数组或者tuple
		# mapping为数组或者字符串，元素以逗号隔开
		super(DBClass, self).__init__()
		self._datas = {}
		names = self.get_names()
		for name in names:
			self._datas[name] = self.__class__.default_values.get(name, None)
		if item is not None:
			assert mapping is not None
			if isinstance(mapping, (str, unicode)):
				mapping = map(lambda x: x.strip(), mapping.split(','))
			update_datas = dict(zip(mapping, list(item)))
			for name in names:
				if name in update_datas:
					value = update_datas[name]
					if value is not None and name in self.__class__.field_parses:
						parser = self.__class__.field_parses[name]
						self._datas[name] = parser.loads(value)
					else:
						self._datas[name] = value

	def dumps(self):
		value = copy.copy(self._datas)
		return value

	def loads(self, datas):
		self._datas.update(datas)

	def __setattr__(self, name, value):
		if name.startswith('_'):
			self.__dict__[name] = value
		else:
			self._datas[name] = value

	def __getattr__(self, name):
		if name.startswith('_'):
			return self.__dict__[name]
		else:
			return self._datas[name]

	def get_names(self, excludes=[]):
		if not isinstance(excludes, list):
			excludes = [excludes]
		fields = map(lambda x: x.strip(), self.__class__.fields.split(','))
		return list(set(fields) - set(excludes))

	def get_values(self, names):
		# 所有names中的元素的值，且是通过转换后的
		values = []
		for name in names:
			value = self._datas[name]
			if value is not None and name in self.__class__.field_parses:
				value = self.__class__.field_parses[name].dumps(value)
			values.append(value)
		return values

	def get_names_values(self, excludes=[]):
		names = self.get_names(excludes)
		values = self.get_values(names)
		return names, values

	def __repr__(self):
		names, values = self.get_names_values()
		data = dict(zip(names, values))
		return json.dumps(data, indent=4)


class DBClassBaseLoad(object):
	@staticmethod
	def dumps(obj):
		return obj

	@staticmethod
	def loads(obj):
		return obj


class utf8_json(DBClassBaseLoad):
	@staticmethod
	def dumps(obj):
		return json.dumps(obj, ensure_ascii=False)

	@staticmethod
	def loads(obj):
		return json.loads(obj)


class str_load(DBClassBaseLoad):
	@staticmethod
	def loads(obj):
		return str(obj)



if __name__ == '__main__':
	a = DBClass()
	print(a)








