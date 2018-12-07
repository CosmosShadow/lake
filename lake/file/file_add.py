# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import csv
import json
from .file_op import *


def add_csv(row, path):
	with file(path, 'ab') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(row)


def add_txt(data, path):
	with open(path, "a") as text_file:
		if isinstance(data, str):
			text_file.write(str(data))
		elif isinstance(data, list):
			for line in data:
				if isinstance(line, str):
					text_file.write(str(line) + '\n')
		else:
			text_file.write(json.dumps(data))

def add_line(line, path):
	with open(path, "a") as text_file:
		text_file.write(str(line) + '\n')


def new_adder(save_path, create_new=False):
	# 产生一个添加器
	if create_new:
		rm(save_path)
	def add_line(*args):
		for x in args:
			add_txt(x, save_path)
		add_txt('\n', save_path)
	return add_line