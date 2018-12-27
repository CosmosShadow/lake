# coding: utf-8
# Author: wanhui0729@gmail.com
import argparse
import os
import lake.file
import lake.dir
import json
from recordtype import recordtype
import torch


def optionParser(net_option, option_path, args, unknown):
	"""description
	Args:
		net_option(net_option class): 网络选项类
		option_path(str): 配置文件地址
		args(argparse.Namespace): 命令行参数
	Return:
		opt(argparse.Namespace):生成所有选项参数
	优先级：
		1.使用命令行参数
		2.使用文件保存的参数
		3.默认: net_option中默认参数
	"""

	if os.path.exists(option_path):
		option_json = lake.file.read(option_path)
		option_dict = json.loads(option_json)
		# # 命令行参数覆盖
		# print(unknown)
		# exit()
		# if len(unknown) > 0:
		# 	for unknown_arg in unknown:
		# 		key = unknown_arg.split('=')[0][2:]
		# 		value = unknown_arg.split('=')[1]
		# 		option_dict[key] = value
		# 文件转化为配置参数
		opt = recordtype('X', option_dict.keys())(*option_dict.values())
		print('从{}加载option'.format(option_path))
	else:
		opt = net_option()
		opt.option_name = args.option
		set_num_workers(opt)
		option_json = json.dumps(vars(opt), indent=4)
		lake.file.write(option_json, option_path)
		print('从option_{}加载option'.format(opt.option_name))

	return opt


# 配置数据加载线程
def set_num_workers(option):
	gpuCount = torch.cuda.device_count()
	if option.num_workers == 0:
		option.num_workers = gpuCount * 2


if __name__ == '__main__':
	pass










