# coding: utf-8
import csv
import lake
import json
import os


def write(data, path):
	lake.dir.mk(path)
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












