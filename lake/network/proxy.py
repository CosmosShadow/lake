# coding: utf-8

def set_socks(ip, port):
	import socks
	import socket
	import ssl
	ssl._create_default_https_context = ssl._create_unverified_context
	socks.setdefaultproxy(socks.PROXY_TYPE_SOCKS5, ip, port)
	socket.socket = socks.socksocket