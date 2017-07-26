# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function
from .base import *
from .unet import *
from .cnn import *
from .instance_normalization import *
from lake.decorator import singleton


@singleton
class GPU_IDS(object):
	def __init__(self, gpu_ids):
		self.gpu_ids = gpu_ids

def set_default_gpu_ids(gpu_ids):
	GPU_IDS(gpu_ids)

def get_default_gpu_ids():
	return GPU_IDS([]).gpu_ids