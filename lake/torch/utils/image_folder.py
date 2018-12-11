# coding: utf-8
import torch.utils.data as data
from PIL import Image
import os
import os.path
import lake
import numpy as np

def make_dataset(dir):
	if dir.startswith('~'):
		dir = dir.replace('~', os.environ['HOME'])
	images = []
	assert os.path.isdir(dir), '%s is not a valid directory' % dir

	for root, _, fnames in sorted(os.walk(dir)):
		for fname in fnames:
			if lake.image.is_image_file(fname):
				path = os.path.join(root, fname)
				images.append(path)

	return images


def default_loader(path):
	return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset, lake.array.CountNextInterface):
	def __init__(self, root, transform=None, return_paths=False, loader=default_loader):
		imgs = make_dataset(root)
		if len(imgs) == 0:
			raise(RuntimeError("Found 0 images in: " + root))
		self.imgs = imgs
		self.transform = transform
		self.return_paths = return_paths
		self.loader = loader
		self.index = 0

	def __getitem__(self, index):
		path = self.imgs[index]
		img = self.loader(path)
		if self.transform is not None:
			img = self.transform(img)
		if self.return_paths:
			return img, path
		else:
			return img

	def __len__(self):
		return len(self.imgs)

	def _shuffle(self):
		perm = np.arange(len(self))
		np.random.shuffle(perm)
		self.imgs = [self.imgs[j] for j in perm]

	def next(self, batch_size):
		assert batch_size <= len(self)
		if self.index + batch_size > len(self):
			self.index = 0
			self._shuffle()

		imgs = []
		paths = []
		for i in range(self.index, self.index+batch_size):
			if self.return_paths:
				img, path = self[i]
				paths.append(path)
			else:
				img = self[i]
			imgs.append(img)

		self.index = self.index + batch_size
		if self.return_paths:
			return imgs, paths
		else:
			return imgs

	def count(self, batch_size=None):
		if batch_size is None:
			return len(self) / batch_size
		else:
			return len(self)





