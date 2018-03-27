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
	data_q = ms.Queue()
	return_q = ms.Queue()

	for input_data in inputs:
		data_q.put(input_data)

	thread_count = len(locals)
	ps = []
	for i in range(thread_count):
		p = ms.Process(target=target, args=(data_q, return_q, locals[i]))
		ps.append(p)

	for p in ps:
		p.start()
	for p in ps:
		p.join()

	datas = []
	while not return_q.empty():
		datas.append(return_q.get())
	return datas


if __name__ == '__main__':
	# 使用方式如下
	def target(input_q, output_q, local):
		while not input_q.empty():
			data = input_q.get()
			output_q.put(data + local)
			# time.sleep(0.1)

	locals = [2, 3, 5]
	inputs = range(100)
	outputs = pool_local(locals, target, inputs)	
	print outputs








