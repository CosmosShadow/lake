# coding: utf-8
from PIL import Image
import lake
import  os
import sys
import numpy as np
import collections

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

def open_image(path):
	"""打开图片"""
	return Image.open(path).convert('RGB')


def clear_dir(images_dir, debug = True):
	"""清理目录下破损图片"""
	images_path = lake.dir.loop(images_dir, IMG_EXTENSIONS)
	count = len(images_path)
	for i, image_path in enumerate(images_path):
		if debug:
			sys.stdout.write('\r%d/%d' % (i, count))
			sys.stdout.flush()
		try:
			open_image(image_path)
		except Exception as e:
			print(image_path)
			print('')
			os.remove(image_path)


def rotate_with_background(img, angel, bg=(255,)*4):
	"""带背景的旋转
	Args:
		img: PIL image
		angel: rotate angel
		bg: background, can be gray value or RGBA
	Returns:
		rotated image
	"""
	assert isinstance(bg, int) or (isinstance(bg, collections.Iterable) and len(bg) == 4)
	if isinstance(bg, int):
		bg = (bg, ) * 4
	# converted to have an alpha layer
	im2 = img.convert('RGBA')
	# rotated image
	rot = im2.rotate(angel, resample=Image.BICUBIC, expand=True)
	# a white image same size as rotated image
	fff = Image.new('RGBA', rot.size, bg)
	# create a composite image using the alpha layer of rot as a mask
	out = Image.composite(rot, fff, rot)
	# converting back to mode
	return out.convert(img.mode)