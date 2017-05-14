# coding: utf-8
from base import *
from unet import *
from cnn import *
from instance_normalization import *
from lake.decorator.singleton import singleton


@singleton
class GPU_IDS(object):
	def __init__(self, gpu_ids):
		self.gpu_ids = gpu_ids

def set_default_gpu_ids(gpu_ids):
	GPU_IDS(gpu_ids)

def get_default_gpu_ids():
	return GPU_IDS([]).gpu_ids