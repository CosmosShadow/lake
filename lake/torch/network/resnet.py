# coding: utf-8
import torch
import torch.nn as nn

class ResnetBlock(nn.Module):
	def __init__(self, dim, padding_type, norm_layer, use_dropout):
		super(ResnetBlock, self).__init__()
		conv_block = []
		conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=1), norm_layer(dim), nn.ReLU(True)]
		if use_dropout:
			conv_block += [nn.Dropout(0.5)]
		conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=1), norm_layer(dim)]
		self.model = nn.Sequential(*conv_block)

	def forward(self, x):
		out = x + self.model(x)
		return out



