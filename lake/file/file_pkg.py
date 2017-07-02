# coding: utf-8
import sys

def load_pkg(dir_path, pkg_name):
	sys.path.append(dir_path)
	return __import__(pkg_name)