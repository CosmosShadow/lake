# coding: utf-8
import os
import errno
import shutil


def mk(path):
	try:
		os.makedirs(path)
	# Python >2.5
	except OSError as exc:  # pragma: no cover
		if exc.errno == errno.EEXIST and os.path.isdir(path):
			pass
		else:
			raise

def rm(path):
	if os.path.exists(path):
		shutil.remove(path)

def remk(path):
	if os.path.exists(path):
		shutil.rmtree(path)
	mk(path)


def check_dir(fname):
	dname = os.path.dirname(fname)
	if dname not in set([".", "..", ""]):
		mk(dname)

