# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function
from .image_numpy_op import *
from .image_pil_op import *

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

def is_image_file(filename):
	return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)