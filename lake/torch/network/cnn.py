# coding: utf-8
import torch
import torch.nn as nn
import lake

def gen_cnn(n_layers, nfs, kernels, strides, pads, norm=None):
	"""生成CNN
	Args:
		n_layers : 网络层数
		nfs: array or int, 卷积特征大小
		strides: array or int, 卷积核大小
		pads: array or int, padding大小
		norm: nn.BatchNorm2d, ...
	Returns:
		b : type
	"""
	nfs = lake.array.extend(nfs, n_layers+1)
	kernels = lake.array.extend(kernels, n_layers)
	strides = lake.array.extend(strides, n_layers)
	pads = lake.array.extend(pads, n_layers)

	sequence = []
	for i in range(n_layers):
		sequence += [nn.Conv2d(nfs[i], nfs[i+1], kernel_size=kernels[i], stride=strides[i], padding=pads[i])]
		if nfs[i] > 3 and norm is not None:
			sequence += [norm(nfs[i+1], affine=True)]
		sequence += [nn.LeakyReLU(0.2, True)]
	return sequence