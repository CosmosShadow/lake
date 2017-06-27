# coding: utf-8
from PIL import Image
import lake
import  os
import sys
from scipy.misc import *
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
plt.ioff()

# imread
# imsave
# imresize
# 

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
	return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def open_image(path):
	return Image.open(path).convert('RGB')

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


def clear_dir(images_dir):
	"""清理目录下破损图片"""
	images_path = lake.file.list_dir(images_dir, IMG_EXTENSIONS)
	count = len(images_path)
	for i, image_path in enumerate(images_path):
		sys.stdout.write('\r%d/%d' % (i, count))
		sys.stdout.flush()
		try:
			open_image(image_path)
		except Exception as e:
			print image_path
			print ''
			os.remove(image_path)


def save_hist(data, save_path, bins=None, to_percentage=False):
	"""保存成直方图
	Args:
		data : numpy
		bins: like np.linspace(0, 200, 201)
	"""
	if bins is None:
		bins = np.linspace(data.min(), data.max()+1, 100)
	fig = plt.gcf()
	plt.hist(data, bins=bins, alpha=0.5, label='num', color='red')
	plt.legend(loc='upper right')
	plt.xlim(0,len(bins)-1)
	plt.xlabel('x')
	plt.ylabel('y')

	if to_percentage:
		to_percentage = lambda y, pos: str(round( ( y / float(len(data)) ) * 100.0, 2)) + '%'
		plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percentage))

	fig.savefig(save_path, dpi=200)
	plt.close('all')

if __name__ == '__main__':
	data = [['x', 'y', 'z'], [1, 10, 9], [2, 5, 1], [3, 9, 0]]
	csv_to_image(data, '1.png')

