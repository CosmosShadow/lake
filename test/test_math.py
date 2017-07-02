# coding: utf-8
import lake


class TestMathFun(object):
	@classmethod
	def setup_class(klass):
		print """运行测试前，加载数据"""
 
	@classmethod
	def teardown_class(klass):
		print """运行测试后，释放资源"""
 
	def setup(self):
		print """在每个测试函数运行之前，执行一次"""
 
	def teardown(self):
		print """在每个测试函数运行之后，执行一次"""
 
	def test_PieceLinearFun(self):
		linear = lake.mt.PieceLinearFun(100, 200, 0, 1)
		assert linear(0) == 0
		assert linear(100) == 0
		assert linear(150) == 0.5
		assert linear(200) == 1
		assert linear(300) == 1











