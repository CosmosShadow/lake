# coding: utf-8
import time

def current_date_str():
	time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())