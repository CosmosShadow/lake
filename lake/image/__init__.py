# coding: utf-8

def is_image_file(filename):
	IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
	return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
