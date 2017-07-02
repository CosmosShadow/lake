# coding: utf-8
import csv
import json


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

def add_line(line, path):
	with open(path, "ab") as text_file:
		text_file.write(str(line) + '\n')