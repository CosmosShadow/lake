# coding: utf-8
import torch
import torch.nn as nn
import lake

def gen_cnn(n_layers, nfs, kernels, strides, pads, pools=1, norm=nn.BatchNorm2d, last_activate=True):
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
	pools = lake.array.extend(pools, n_layers)

	sequence = []
	for i in range(n_layers):
		sequence += [nn.Conv2d(nfs[i], nfs[i+1], kernel_size=kernels[i], stride=strides[i], padding=pads[i])]
		if nfs[i] > 3 and norm is not None:
			sequence += [norm(nfs[i+1], affine=True)]
		if i < n_layers - 1 or last_activate:
			sequence += [nn.LeakyReLU(0.2, True)]
		if pools[i] > 1:
			sequence += [nn.MaxPool2d(pools[i])]
	return sequence


def gen_cnn_transpose(n_layers, nfs, kernels, strides, pads, norm=nn.BatchNorm2d, last_activate=True):
	"""生成反向CNN
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
	for i in range(n_layers-1, -1, -1):
		sequence += [nn.ConvTranspose2d(nfs[i+1], nfs[i], kernel_size=kernels[i], stride=strides[i], padding=pads[i])]
		if nfs[i-1] > 3 and norm is not None:
			sequence += [norm(nfs[i], affine=True)]
		if i > 0 or last_activate:
			sequence += [nn.LeakyReLU(0.2, True)]
	return sequence













