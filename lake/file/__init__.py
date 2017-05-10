# coding: utf-8
import csv
import lake
import json

def write(data, path):
	lake.dir.check_dir(path)
	if path.endswith('.csv'):
		wirte_csv(data, path)
	else:
		with open(path, "w") as text_file:
			if isinstance(data, str):
				text_file.write(str(data))
			else:
				text_file.write(json.dumps(data))


def read(path):
	if path.endswith('.csv'):
		return read_csv(path)
	else:
		with open(path, "r") as text_file:
			return text_file.readlines()


def wirte_csv(data, path):
	with file(path, 'wb') as csvfile:
		writer = csv.writer(csvfile)
		for row in data:
			writer.writerow(row)


def read_csv(path, delimiter=','):
	with file(path, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=delimiter)
		data = [line for line in reader]
	return data


if __name__ == '__main__':
	import os
	write('12q3q32', 'tmp/tmp/1.txt')
	print read('tmp/tmp/1.txt')
	os.system('rm -rf tmp')












