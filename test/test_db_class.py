# coding: utf-8
from nose.tools import *
import json
from lake.data.db_class import DBClass


class SubClass(DBClass):
	fields = 'id, contents, status'
	field_parses = {
		'contents': json,
	}
	default_values = {
		'status': 1
	}

class Test(object):
	def setup(self):
		self.contents = [1, 2, 3]
		contents_str = json.dumps(self.contents)
		item = (2, contents_str)
		mapping = 'id, contents'		# 或 ['id', 'contents']
		self.obj = SubClass(item, mapping)

	def test_default(self):
		a = SubClass()
		assert_equal(a.status, 1)

	def test_parse(self):
		assert_equal(self.obj.id, 2)
		assert_equal(self.obj.contents, self.contents)

	def test_get_names_values(self):
		names, values = self.obj.get_names_values()
		assert_equal(names, ['status', 'id', 'contents'])
		assert_equal(values, [1, 2, '[1, 2, 3]'])



if __name__ == '__main__':
	test = Test()
	test.setup()
	test.test_default()
	test.test_parse()
	test.test_get_names_values()











