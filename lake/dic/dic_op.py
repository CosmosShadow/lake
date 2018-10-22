# coding: utf-8

def reverse_list_v(values):
	"""反转dict, dict中的元素为数组
	Args:
		values: {k1: [v1, v2, ...], ...}
	Returns:
		{v1: k1, v2: k2, ...}
	"""
	return dict([(item, i) for i, j in values.items() for item in j])