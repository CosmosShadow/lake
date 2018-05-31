# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function
from .downloader import download
from . import proxy
import lake.file

import requests

def download_file(url, save_path, headers={}, proxies=None, max_size=None, timeout=None):
	"""下载文件: 主要是大文件，流式下载
	Args:
		url: 下载地址
		save_path: 保存地址
		headers: 请求头
		proxies: 代理
		max_size: 最大大小，超过则下载失败，防止下载过多
	Returns:
		bool: 是否下载成功
	"""
	size = 0
	chunk_size = 2048
	r = requests.get(url, stream=True, headers=headers, proxies=proxies)
	with open(save_path, 'wb') as fd:
		for chunk in r.iter_content(chunk_size):
			fd.write(chunk)
			size += chunk_size
			if max_size is not None and size > max_size:
				lake.file.rm(save_path)
				return False
	return True
