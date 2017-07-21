# coding: utf-8

def continue_divide(value, div, max_left=1):
	"""连续除"""
	assert div > 1
	assert max_left >= 1
	assert isinstance(value, int)
	assert isinstance(div, int)
	assert isinstance(max_left, int)
	ex = 0
	left = value
	while left > max_left:
		left = left / div
		ex += 1
	return ex, left