# coding: utf-8
import lake
import nltk
from nltk.draw.util import CanvasFrame
from nltk.draw import TreeWidget

def save_tree(str_tree, save_path):
	# 输出可视化结果
	cf = CanvasFrame()
	tc = TreeWidget(cf.canvas(), str_tree)
	tc['node_font'] = 'arial 14 bold'
	tc['leaf_font'] = 'arial 14'
	tc['node_color'] = '#005990'
	tc['leaf_color'] = '#3F8F57'
	tc['line_color'] = '#175252'
	cf.add_widget(tc, 10, 10) # (10,10) offsets
	cf.print_to_file('tmp.ps')
	cf.destroy()

	# 使用ImageMagick工具进行转换
	lake.shell.run('convert tmp.ps %s' % save_path)
	lake.shell.run('rm tmp.ps')

def save_tree_with_ps(str_tree, save_path):
	# 输出可视化结果
	cf = CanvasFrame()
	tc = TreeWidget(cf.canvas(), str_tree)
	tc['node_font'] = 'arial 14 bold'
	tc['leaf_font'] = 'arial 14'
	tc['node_color'] = '#005990'
	tc['leaf_color'] = '#3F8F57'
	tc['line_color'] = '#175252'
	cf.add_widget(tc, 10, 10) # (10,10) offsets
	cf.print_to_file('{}.ps'.format(save_path))
	cf.destroy()

