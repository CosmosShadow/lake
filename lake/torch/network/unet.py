# coding: utf-8
import torch
import torch.nn as nn
from lake.torch.network.base import Base


class Unet(Base):
	def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
		super(Unet, self).__init__(gpu_ids)
		assert(input_nc == output_nc)

		unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, innermost=True)
		for i in range(num_downs, 0, -1):
			unet_block = UnetSkipConnectionBlock(ngf * min(8, 2**(i-1)), ngf * min(8, 2**i), unet_block)
		unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True)

		self.model = unet_block


class UnetSkipConnectionBlock(nn.Module):
	def __init__(self, outer_nc, inner_nc, submodule=None, outermost=False, innermost=False, use_dropout=False):
		super(UnetSkipConnectionBlock, self).__init__()
		self.outermost = outermost

		downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4, stride=2, padding=1)
		downrelu = nn.LeakyReLU(0.2, True)
		downnorm = nn.BatchNorm2d(inner_nc)
		uprelu = nn.ReLU(True)
		upnorm = nn.BatchNorm2d(outer_nc)

		if outermost:
			upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
			down = [downconv]
			up = [uprelu, upconv, nn.Tanh()]
			model = down + [submodule] + up
		elif innermost:
			upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1)
			down = [downrelu, downconv]
			up = [uprelu, upconv, upnorm]
			model = down + up
		else:
			upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
			down = [downrelu, downconv, downnorm]
			up = [uprelu, upconv, upnorm]

			if use_dropout:
				model = down + [submodule] + up + [nn.Dropout(0.5)]
			else:
				model = down + [submodule] + up

		self.model = nn.Sequential(*model)

	def forward(self, x):
		if self.outermost:
			return self.model(x)
		else:
			return torch.cat([self.model(x), x], 1)


