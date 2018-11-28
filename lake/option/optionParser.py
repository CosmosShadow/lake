# coding: utf-8
# Author: wanhui0729@gmail.com
import argparse
import os
import lake.file
import lake.dir
import json
from recordtype import recordtype

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
        # 命令行参数覆盖
        if len(unknown) > 0:
            net_option_dict = net_option().__dict__
            for unknown_arg in unknown:
                unknown_arg_key = unknown_arg.split('=')[0][2:]
                option_dict[unknown_arg_key] = net_option_dict[unknown_arg_key]
        # 文件转化为配置参数
        opt = recordtype('X', option_dict.keys())(*option_dict.values())
        print('从{}加载option'.format(option_path))
    else:
        opt = net_option()
        opt.option_name = args.option
        option_json = json.dumps(vars(opt), indent=4)
        lake.file.write(option_json, option_path)
        print('从option_{}加载option'.format(opt.option_name))

    return opt


if __name__ == '__main__':
    pass
