# coding: utf-8
import torch
import torch.nn as nn
import lake
from .base import Base


class Unet(Base):
	def __init__(self, n_layers, nfs, norm=nn.BatchNorm2d, dropout=False):
		super(Unet, self).__init__()
		assert n_layers >= 3
		nfs = lake.array.extend(nfs, n_layers)

		unet_block = UnetSkipConnectionBlock(nfs[-1], nfs[-1], innermost=True, norm=norm, dropout=dropout)
		for i in range(1, n_layers-1):
			unet_block = UnetSkipConnectionBlock(nfs[-i-1], nfs[-i], unet_block, norm=norm, dropout=dropout)
		unet_block = UnetSkipConnectionBlock(nfs[0], nfs[1], unet_block, outermost=True, norm=norm, dropout=dropout)

		self.model = unet_block


class UnetSkipConnectionBlock(nn.Module):
	def __init__(self, outer_nc, inner_nc, submodule=None, outermost=False, innermost=False, norm=nn.BatchNorm2d, dropout=False):
		super(UnetSkipConnectionBlock, self).__init__()
		self.outermost = outermost

		downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4, stride=2, padding=1)
		downrelu = nn.LeakyReLU(0.2, True)
		downnorm = norm(inner_nc, affine=True)
		uprelu = nn.ReLU(True)
		upnorm = norm(outer_nc, affine=True)

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

			if dropout:
				model = down + [submodule] + up + [nn.Dropout(0.5)]
			else:
				model = down + [submodule] + up

		self.model = nn.Sequential(*model)

	def forward(self, x):
		if self.outermost:
			return self.model(x)
		else:
			return torch.cat([self.model(x), x], 1)


if __name__ == '__main__':
	net = Unet(3, [3, 64, 128])
	net = Unet(5, [3, 64, 128])
	net.print_net()








