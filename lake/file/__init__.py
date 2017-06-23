# coding: utf-8
import csv
import lake
import json
from loop_file import *


def write(data, path):
	lake.dir.check_dir(path)
	if path.endswith('.csv'):
		write_csv(data, path)
	elif path.endswith('.dot'):
		write_dot(data, path)
	else:
		write_txt(data, path)


def write_txt(data, path):
	with open(path, "w") as text_file:
		if isinstance(data, str):
			text_file.write(str(data))
		elif isinstance(data, list):
			for line in data:
				if isinstance(line, str):
					text_file.write(str(line) + '\n')
		else:
			text_file.write(json.dumps(data))


def write_csv(data, path):
	with file(path, 'wb') as csvfile:
		writer = csv.writer(csvfile)
		for row in data:
			writer.writerow(row)


def write_dot(data, path):
	dot = "digraph G {\n"
	for item in data:
		if len(item) == 2:
			start, end = item[0], item[1]
			dot += '    "%s"->"%s";\n' % (start, end)
		if len(item) == 3:
			start, end, color = item[0], item[1], item[2]
			dot += '    "%s"->"%s" [color="%s"];\n' % (start, end, color)
	dot += "}"
	write_txt(dot, path)


def read(path):
	if path.endswith('.csv'):
		return read_csv(path)
	elif path.endswith('.json'):
		with open(path, "r") as text_file:
			return ''.join(text_file.readlines())
	else:
		with open(path, "r") as text_file:
			return [line.strip() for line in text_file.readlines()]


def read_csv(path, delimiter=','):
	with file(path, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=delimiter)
		data = [line for line in reader]
	return data


def add_csv(row, path):
	with file(path, 'ab') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(row)


def add_txt(data, path):
	with open(path, "ab") as text_file:
		if isinstance(data, str):
			text_file.write(str(data))
		elif isinstance(data, list):
			for line in data:
				if isinstance(line, str):
					text_file.write(str(line) + '\n')
		else:
			text_file.write(json.dumps(data))


if __name__ == '__main__':
	import os
	write('12q3q32', 'tmp/tmp/1.txt')
	print read('tmp/tmp/1.txt')
	os.system('rm -rf tmp')

	add_csv([1, 2, 3], '1.csv')












