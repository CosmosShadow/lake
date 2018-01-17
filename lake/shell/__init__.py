# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import signal
import subprocess
import lake
import time


class Shell(object):
	def __init__(self, cmd, out='print', sleep=None):
		super(Shell, self).__init__()
		self._cmd = cmd
		self._out = out
		self._sleep = sleep or 1	#起码等待1s
		
	def __enter__(self):
		# print('Shell enter: ', self._cmd)
		self.pro = subprocess.Popen(self._cmd, stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
		time.sleep(self._sleep)


	def __exit__(self, exc_type, exc_value, exc_tb):
		# self.pro.terminate()
		os.killpg(os.getpgid(self.pro.pid), signal.SIGTERM)
		output = self.pro.stdout.read()
		if self._out == 'print':
			print(output)
		elif self._out is not None:
			lake.file.write(output, self._out)
		# print('Shell exit: ', self._cmd)


class MC(object):
	"""docstring for MC"""
	def __init__(self, port, size='100m'):
		super(MC, self).__init__()
		self._cmd = 'memcached -d -m %s -p %d' % (size, port)

	def __enter__(self):
		# print('memcache start: ', self._cmd)
		os.system(self._cmd)
		time.sleep(1)			#等待启动完成

	def __exit__(self, exc_type, exc_value, exc_tb):
		cmd = '''ps -ef | grep '%s' | awk '{print $2}' | xargs kill -9''' % self._cmd
		os.system(cmd)
		# print('memcache stop: ', self._cmd)


def run(cmd):
	"""运行返回结果"""
	output = os.popen(cmd)
	return output.read()


def run_with_timeout(cmd, timeout_sec):
	"""有时限的行运脚本
	Args:
		cmd : 命令行
		timeout_sec: 超时时长(秒为单位)
	"""
	import subprocess
	from threading import Timer
	proc = subprocess.Popen(cmd, shell=True)
	kill_proc = lambda p: p.kill()
	timer = Timer(timeout_sec, kill_proc, [proc])
	try:
		timer.start()
		stdout,stderr = proc.communicate()
	finally:
		timer.cancel()



def call(cmd):
	"""运行不返回结果"""
	os.system(cmd)