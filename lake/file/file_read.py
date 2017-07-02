# coding: utf-8
import csv
import json
import os

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











