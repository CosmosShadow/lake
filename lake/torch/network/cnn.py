# coding: utf-8
import torch
import torch.nn as nn
import lake

def gen_cnn(n_layers, nfs, kernels, strides, pads, norm=BatchNorm2d):
	nfs = lake.array.extend(nfs, n_layers+1)
	kernels = lake.array.extend(kernels, n_layers)
	strides = lake.array.extend(strides, n_layers)
	pads = lake.array.extend(pads, n_layers)

	sequence = []
	for i in range(n_layers):
		sequence += [nn.Conv2d(nfs[i], nfs[i+1], kernel_size=kernels[i], stride=strides[i], padding=pads[i])]
		if nfs[i] > 3:
			sequence += [norm(nfs[i+1], affine=True)]
		sequence += [nn.LeakyReLU(0.2, True)]
	return sequence