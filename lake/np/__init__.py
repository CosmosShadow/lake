# coding: utf-8
import numpy as np


def one_hot(num, index):
	base = np.zeros(num)
	if 0 <= index < num:
		base[index] = 1
	return base


if __name__ == '__main__':
	print one_hot(10, 1)