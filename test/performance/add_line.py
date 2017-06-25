# coding: utf-8

# 100000次写入，时间分别如下，结果无太大差别
# ==> time-cost: 3.956674     one_by_one
# ==> time-cost: 3.775688     all

count = 100000
text = 'hello  world'
import lake

@lake.decorator.time_cost
def one_by_one():
	for _ in range(count):
		lake.file.add_line(text, 'one_by_one.txt')

@lake.decorator.time_cost
def all():
	for _ in range(count):
		with open('all.txt', "ab") as text_file:
			text_file.write(text + '\n')

one_by_one()
all()

lake.file.rm('one_by_one.txt')
lake.file.rm('all.txt')