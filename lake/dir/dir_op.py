# coding: utf-8
import os
import errno
import shutil

def rm(path):
	if os.path.exists(path):
		shutil.remove(path)

def mk(fname):
	dname = os.path.dirname(fname)
	if dname not in set([".", "..", ""]):
		if not op.path.isdir(dname):
			os.makedirs(dname)

def remk(path):
	rm(path)
	mk(path)