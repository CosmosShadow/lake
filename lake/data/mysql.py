# coding: utf-8
import sys
import os
import logging
import logging.config
import time
import MySQLdb
from MySQLdb import cursors
from DBUtils.PooledDB import PooledDB
import lake


_logger = logging.getLogger(__name__)

def execute_with_log(self, sql, data=()):
	"""
	带日志输出的SQL执行。改造了Cursor类，自动输出日志，方便Debug和查问题。
	"""
	start_time = time.time()
	_logger.info(sql)
	# print data
	self.execute(sql, data)
	end_time = time.time()

	# 神奇，self._last_executed资源可能被干掉了
	last_executed = ''
	if hasattr(self, '_last_executed'):
		last_executed = self._last_executed[:1000]
		if end_time - start_time < 1.0:
			_logger.info('[SQL %i ms]\n\033[0;32m%s\033[0m', (end_time - start_time) * 1000, last_executed)
		else:
			_logger.info('\033[0;31m[SQL %i ms]\n%s\033[0m', (end_time - start_time) * 1000, last_executed)


@lake.decorator.singleton
class MySQLClient(object):
	"""
	MySQL客户端封装。
	"""
	def __init__(self, conf, pool_count=5):
		"""
		Args:
			conf : 数据连接配制
			pool_count: 连接池连接数
		Returns:
			b : type
		"""
		# 增加execute_with_log方法，只需执行一次
		cursors.Cursor.execute_with_log = execute_with_log
		try:
			self._pool = PooledDB(MySQLdb, pool_count, **conf)
		except Exception as e:
			logging.exception(e)
			exit()

	def _get_connect_from_pool(self):
		return self._pool.connection()


	def execute(self, sql, data):
		"""
		执行SQL
		Args:
			sqls: 数组，元素是SQL字符串
		Returns:
			None
		"""
		insert_id = None
		try:
			conn = self._get_connect_from_pool()
			cursor = conn.cursor()
			cursor.execute_with_log(sql, data)
			conn.commit()
			insert_id = cursor.lastrowid
		except Exception as e:
			logging.exception(e)
			conn.rollback()
			raise
		finally:
			cursor.close()
			conn.close()
		return insert_id


	def executemany(self, sql, datas):
		"""
		Args: 
			sql: SQL语句
			datas: 数据
		Returns:
			None
		"""
		try:
			conn = self._get_connect_from_pool()
			cursor = conn.cursor()
			cursor.executemany(sql, datas)
			conn.commit()
		except Exception as e:
			logging.exception(e)
			conn.rollback()
		finally:
			cursor.close()
			conn.close()


	def fetchall(self, sql, data=()):
		"""
		查询SQL，获取所有指定的列
		Args: 
			sql: SQL语句
		Returns:
			结果集
		"""
		try:
			conn = self._get_connect_from_pool()
			cursor = conn.cursor()
			cursor.execute_with_log(sql,data)
			results = cursor.fetchall()
			return results
		except Exception as e:
			logging.exception(e)
			raise e
		finally:
			cursor.close()
			conn.close()
		return []


class DBTable(object):
	"""数据库表操作: 拉取、推送
		可用于数据表备份、同步
	"""
	def __init__(self, config):
		super(DBTable, self).__init__()
		# host, port, user, password, db
		self.host = config.get('host', 'localhost')
		self.port = config.get('port', 3306)
		self.user = config.get('user', 'root')
		self.password = config.get('passwd')
		self.db = config.get('db')

	def pull(self, table, save_path='tmp.sql'):
		shell = 'source ~/.bash_profile & mysqldump --lock-tables=false --set-gtid-purged=OFF -h%s -P%d -u%s -p%s %s %s > %s' % \
			(self.host,  self.port, self.user, self.password, self.db, table, save_path)
		lake.shell.call(shell)

	def push(self, sql_path='tmp.sql'):
		shell = 'source ~/.bash_profile & mysql -h%s -P%s -u%s -p%s %s < %s' % (self.host, self.port, self.user, self.password, self.db, sql_path)








