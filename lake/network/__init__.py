# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function
from .downloader import download
from . import proxy

import requests

def download_file(url, save_path, proxies=None):
	"""下载文件: 主要是大文件，流式下载"""
	chunk_size = 128
	r = requests.get(url, stream=True, proxies=proxies)
	with open(save_path, 'wb') as fd:
		for chunk in r.iter_content(chunk_size):
			fd.write(chunk)