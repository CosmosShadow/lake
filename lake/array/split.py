# coding: utf-8
# Author: lichenarthurdata@gmail.com
# 分割
import numpy as np

def split_index(count, rates):
	np.random.seed(1)
	indices = list(range(count))
	np.random.shuffle(indices)

	datas = []
	start = 0
	for rate in rates:
		step = int(rate * count)
		data = indices[start: start + step]
		start += step
		datas.append(data)

	return tuple(datas)