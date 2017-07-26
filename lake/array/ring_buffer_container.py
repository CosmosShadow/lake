# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function
from .ring_buffer import RingBuffer
from .count_next_interface import CountNextInterface
from .sample_ import sample_in_range


class RingBufferContainer(CountNextInterface):
	"""RingBuffer容器，管理多个循环缓存
	-----------------------------------------------------Waring-----------------------------------------------------------
	read_masks: [(-3, 5), 1, 1, 1]
	如果长度只有5，且每个元素等于位置，则
	返回的是([[1, 2, 3, 4, 5]], [4], [4], [4])
	-----------------------------------------------------Waring-----------------------------------------------------------
	"""
	def __init__(self, max_len, buffer_count, read_masks=None):
		"""初始化
		Args:
			max_len: 每个缓存的长度
			buffer_count: 缓存器个数
			read_masks: 读取标识数组，值为数字(表示取多少个)、tuple(开始的offset，个数)，详见next函数
		"""
		assert buffer_count > 0
		self._max_len = max_len
		self._buffer_count = buffer_count
		self._read_masks = read_masks or [1] * buffer_count
		assert len(self._read_masks) == buffer_count
		self._datas = [RingBuffer(max_len) for _ in range(buffer_count)]

	def __len__(self):
		return len(self._datas[0])

	def append(self, values):
		assert len(values) == self._buffer_count
		for i, value in enumerate(values):
			self._datas[i].append(value)

	def _format_read_masks(self, read_masks):
		"""根据读取的mask，解析能够采样数据的范围
		return:
			sample_range: (start, end)，不包含结尾
		"""
		new_masks = []
		for read_mask in read_masks:
			if isinstance(read_mask, int):
				new_masks.append((0, read_mask))
			elif isinstance(read_mask, tuple) and len(read_mask) == 2:
				offset, count = read_mask
				assert count > 0
				new_masks.append((offset, offset+count))
			else:
				raise ValueError('read musk is not int or tuple of 2')
		begin_offset = min([begin for begin, end in new_masks])
		end_offset = max([end for begin, end in new_masks])
		sample_range = (max(0, -begin_offset), min(len(self), len(self) - end_offset + 1))
		return sample_range, new_masks

	def next(self, batch_size, read_masks=None):
		read_masks = read_masks or self._read_masks
		assert len(read_masks) == self._buffer_count
		sample_range, new_masks = self._format_read_masks(read_masks)
		idxes = sample_in_range(sample_range[0], sample_range[1], batch_size)
		results = []
		for i in range(self._buffer_count):
			result = []
			begin_offset, end_offset = new_masks[i]
			for idx in idxes:
				if end_offset - begin_offset == 1:
					result.append(self._datas[i][idx+begin_offset])
				else:
					result.append(self._datas[i][idx+begin_offset: idx+end_offset])
			results.append(result)
		return tuple(results)

	def count(self, batch_size=None):
		if batch_size is None:
			return len(self)
		else:
			return len(self) / batch_size
