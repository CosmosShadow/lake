# coding: utf-8
import csv

def write(data, path):
	if path.endswith('.csv'):
		wirte_csv(data, path)


def read(path):
	if path.endswith('.csv'):
		read_csv(path)


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