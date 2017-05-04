# coding: utf-8
import numpy as np


def one_hot(num, index):
	base = np.zeros(num)
	if index >= 0 and index < num:
		base[index] = 1
	return base


if __name__ == '__main__':
	print one_hot(10, 1)