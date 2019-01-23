# coding: utf-8
import multiprocessing as ms

def pool_local(locals, target, inputs):
	"""带local变量的pool, target只运行一遍, 函数参数为(data_q, return_q, locals[i])
	Args:
		locals : 传给每个线程的局部变量
		target: 目标函数
		inputs: 并发处理的目标
	Returns:
		target保存在队列的数据
	"""
	thread_count = len(locals)

	data_q = ms.Queue(thread_count)
	return_q = ms.Queue()
	
	ps = []
	for i in range(thread_count):
		p = ms.Process(target=target, args=(data_q, return_q, locals[i]))
		p.start()
		ps.append(p)

	print('all started')

	for input_data in inputs:
		data_q.put(input_data)
	for _ in range(thread_count):
		data_q.put('stop')
	print('put data finish')

	datas = []
	for _ in range(thread_count):
		print('index: %d start get' % i)
		datas += return_q.get()

	for i, p in enumerate(ps):
		print('index: %d start join' % i)
		p.join()

	return datas


if __name__ == '__main__':
	# 使用方式如下
	def target(input_q, output_q, local):
		results = []
		img_path = local
		m = Match(img_path)
		while True:
			try:
				image_path = input_q.get(timeout=10.0)
			except Exception as e:
				print(e)
				break
			if image_path == 'stop':
				print('get stop')
				break
			try:
				code, same_count = m.is_contain(image_path)
			except Exception as e:
				print(image_path)
				print(e)
				continue
			print('similary: %d   code: %d  image_path: %s' % (same_count, code, image_path))
			result = (image_path, code, same_count)
			results.append(result)
		output_q.put(results)
		print('inner terminate')








