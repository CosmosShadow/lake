# coding: utf-8
from PIL import Image
import matplotlib.pyplot as plt
import lake

def is_image_file(filename):
	IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
	return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def open(path):
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


if __name__ == '__main__':
	data = [['x', 'y', 'z'], [1, 10, 9], [2, 5, 1], [3, 9, 0]]
	csv_to_image(data, '1.png')