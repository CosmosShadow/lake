# coding: utf-8
import lake


class TestDataChapter(object):
	@classmethod
	def setup_class(klass):
		pass
 
	@classmethod
	def teardown_class(klass):
		pass
 
	def setup(self):
		pass
 
	def teardown(self):
		pass
 
	def test_sample(self):
		count = 100
		percent = 0.2

		data = range(count)
		sampled, left = lake.array.sample(data, percent)

		assert len(sampled) == int(count * percent)
		assert len(left) == int(count * (1-percent))