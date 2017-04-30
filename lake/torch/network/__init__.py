# coding: utf-8
from base import *
from unet import *
from instance_normalization import *

def print_net(net):
	print(net)
	params_count = sum([param.numel() for param in net.parameters()])
	print('Total number of parameters: %d' % params_count)