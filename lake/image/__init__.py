# coding: utf-8
from PIL import Image

def is_image_file(filename):
	IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
	return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def open(path):
	return Image.open(path).convert('RGB')