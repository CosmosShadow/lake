# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import signal
import subprocess
import lake

class Shell(object):
	def __init__(self, cmd, out='print'):
		super(Shell, self).__init__()
		self._cmd = cmd
		self._out = out
		
	def __enter__(self):
		self.pro = subprocess.Popen(self._cmd, stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)

	def __exit__(self, exc_type, exc_value, exc_tb):
		# self.pro.terminate()
		os.killpg(os.getpgid(self.pro.pid), signal.SIGTERM)
		output = self.pro.stdout.read()
		if self._out == 'print':
			print(output)
		elif self._out is not None:
			lake.file.write(output, self._out)

def run(cmd):
	os.system(cmd)