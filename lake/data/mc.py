# coding: utf-8
import sys
import os
import pylibmc
import logging
import time
import lake


@lake.decorator.singleton
class MCClient(object):
	"""
	memcached客户端封装。
	"""
	def __init__(self, hosts, default_expire=3600):
		self._mc = pylibmc.Client(hosts, binary=True, behaviors={"tcp_nodelay": True, "ketama": True})
		self._default_expire = default_expire
		self._logger = logging.getLogger(__name__)

	def __del__(self):
		self._mc.disconnect_all()

	def get(self, key):
		value = None
		try:
			value = self._mc.get(key)
			if value != None:
				self._logger.debug('\n\033[0;32m[memcached GET CACHED] %s\033[0m', key)
		except Exception as e:
			self._logger.error('MCClient error: ', exc_info=True)
		return value

	def set(self, key, value, time_expire=None):
		start = time.time()
		try:
			time_expire = time_expire or self._default_expire
			value = self._mc.set(key, value, time=time_expire)
			self._logger.debug('\n\033[0;32m[memcached SET] %s\033[0m', key)
		except Exception as e:
			self._logger.error('MCClient error: ', exc_info=True)
		
