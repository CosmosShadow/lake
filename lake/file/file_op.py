# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os

def rm(path):
	if os.path.exists(path):
		os.remove(path)