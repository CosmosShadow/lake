# coding: utf-8
import os

def rm(path):
	if os.path.exists(path):
		os.remove(path)