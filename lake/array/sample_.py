# coding: utf-8
import random
import numpy as np
import warnings


def sample(data, sample_rate):
	"""
	Args:
		data : list or numpy
		sample_rate : 
	Returns:
		sampled, left
	"""
	count = len(data)
	perm = np.arange(count)
	np.random.shuffle(perm)

	sample_count = int(count * sample_rate) if isinstance(sample_rate, float) else sample_rate

	assert sample_count > 0
	assert sample_count < count

	sample_index = perm[:sample_count]
	left_index = perm[sample_count:]

	if isinstance(data, list):
		return [data[i] for i in sample_index], [data[i] for i in left_index]
	else:
		return data[sample_index], data[left_index]


def sample_in_range(low, high, size):
	assert low < high
	if high - low >= size:
		r = range(low, high)
		batch_idxs = random.sample(r, size)
	else:
		warnings.warn('Not enough entries to sample without replacement. Consider increasing your warm-up phase to avoid oversampling!')
		batch_idxs = np.random.random_integers(low, high - 1, size=size)
	assert len(batch_idxs) == size
	return batch_idxs























