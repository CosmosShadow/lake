# coding: utf-8
import os
import errno


def mkdir(path):
	try:
		os.makedirs(path)
	# Python >2.5
	except OSError as exc:  # pragma: no cover
		if exc.errno == errno.EEXIST and os.path.isdir(path):
			pass
		else:
			raise


def check_dir(fname):
	dname = os.path.dirname(fname)
	if dname not in set([".", "..", ""]):
		mkdir(dname)

