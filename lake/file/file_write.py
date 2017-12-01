# coding: utf-8
import csv
import lake
import json
import os


def write(data, path):
	lake.dir.mk(path)
	if path.endswith('.csv'):
		write_csv(data, path)
	elif path.endswith('.dot') and isinstance(data, list):
		write_dot(data, path)
	else:
		write_txt(data, path)


def write_txt(data, path):
	with open(path, "w") as text_file:
		if isinstance(data, (str, unicode)):
			text_file.write(str(data))
		elif isinstance(data, list):
			for line in data:
				if isinstance(line, (str, unicode)):
					text_file.write(str(line) + '\n')
		else:
			text_file.write(json.dumps(data))


def write_csv(data, path, is_excel=False):
	with file(path, 'wb') as csvfile:
		if is_excel:
			csvfile.write(u"\ufeff")
		writer = csv.writer(csvfile)
		for row in data:
			# writer.writerow([unicode(s).encode("utf-8") for s in row])
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












