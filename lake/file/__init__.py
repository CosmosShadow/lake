# coding: utf-8
import csv

def write(data, path):
	if path.endswith('.csv'):
		wirte_csv(data, path)


def wirte_csv(data, path):
	with file(path, 'wb') as csvfile:
		writer = csv.writer(csvfile)
		for row in data:
			writer.writerow(row)
