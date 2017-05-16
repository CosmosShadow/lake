# coding: utf-8
from image_folder import ImageFolder
from image import *
from learning_rator import *
import torch.nn as nn


def get_norm_layer(norm_type):
	if norm_type == 'batch':
		norm_layer = nn.BatchNorm2d
	elif norm_type == 'instance':
		norm_layer = nn.InstanceNorm2d
	else:
		print('normalization layer [%s] is not found' % norm)
	return norm_layer