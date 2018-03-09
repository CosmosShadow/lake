# coding: utf-8

import os
import sys
import requests
import socket
from six.moves import queue as Queue
from threading import Thread
import re
import ssl
import shutil



ssl._create_default_https_context = ssl._create_unverified_context


class _download_thread(Thread):
	def __init__(self, queue, timeout, retry):
		Thread.__init__(self)
		self.queue = queue
		self.timeout = timeout
		self.retry = retry

	def run(self):
		while True:
			url, save_path = self.queue.get()
			self.download(url, save_path)
			self.queue.task_done()

	def download(self, url, save_path):
		socket.setdefaulttimeout(self.timeout)

		if not os.path.isfile(save_path):
			print("downloading %s from %s\n" % (save_path, url))
			retry_times = 0
			while retry_times < self.retry:
				try:
					r = requests.get(url, stream=True)
					if r.status_code == 200:
						with open(save_path, 'wb') as f:
							r.raw.decode_content = True
							shutil.copyfileobj(r.raw, f)
							break
				except Exception as e:
					print(e)
				retry_times += 1
			else:
				try:
					os.remove(save_path)
				except OSError:
					pass
				print("Failed to retrieve from %s\n" % (url))


class _downloader(object):
	def __init__(self, thread_count = 10, timeout = 10, retry = 5):
		self.queue = Queue.Queue()
		self.thread_count = thread_count
		self.timeout = timeout
		self.retry = retry
		self.scheduling()

	def scheduling(self):
		for x in range(self.thread_count):
			worker = _download_thread(self.queue, self.timeout, self.retry)
			worker.daemon = True
			worker.start()

	def download(self, download_list):
		# download_list: [[url, save_path], ...]
		for item in download_list:
			self.queue.put((item[0], item[1]))
		self.queue.join()

		print('finish')


def download(download_list, thread_count = 10, timeout = 10, retry = 5):
	# download_list: [(image_url, save_path)]
	op = _downloader(thread_count, timeout, retry)
	op.download(download_list)































