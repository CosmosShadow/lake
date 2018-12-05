# coding: utf-8
import lake
import  os
import sys
import numpy as np

import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
plt.ioff()

from scipy.misc import *
# imread
# imsave
# imresize


def csv_to_image(data, save_path, dpi=200):
	fig = plt.gcf()

	if isinstance(data, str):
		data = lake.file.read(data)

	names = data[0]
	record = data[1:]

	x = [item[0] for item in record]
	plt.xlabel(names[0])
	for i in range(1, len(names)):
		y = [item[i] for item in record]
		plt.plot(x, y, label=names[i])

	plt.legend(names[1:])

	fig.savefig(save_path, dpi=dpi)


def save_hist(data, save_path, bins=None, to_percentage=False, xlabel=None, ylabel=None, title=None):
	"""保存成直方图
	Args:
		data : numpy
		bins: like np.linspace(0, 200, 201)
	"""
	if (xlabel is None) or (ylabel is None):
		xlabel = 'x'
		ylabel = 'y'

	if bins is None:
		bins = np.linspace(data.min(), data.max()+1, 100)

	if title is None:
		title = 'Hist Figure'
	fig = plt.gcf()
	plt.hist(data, bins=bins, alpha=0.5, label='num', color='red')
	plt.legend(loc='upper right')
	plt.xlim(data.min(), data.max())
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)

	if to_percentage:
		to_percentage = lambda y, pos: str(round( ( y / float(len(data)) ) * 100.0, 2)) + '%'
		plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percentage))

	fig.savefig(save_path, dpi=200)
	plt.close('all')


def rgb2gray(rgb):
	return 0.2126 * rgb[..., 0] + 0.0722 * rgb[..., 1] + 0.7152 * rgb[..., 2]


def rgb2y(rgb):
	return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

