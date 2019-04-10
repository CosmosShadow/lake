# 打包
# 参考: https://packaging.python.org/tutorials/distributing-packages/#uploading-your-project-to-pypi

# 清掉文件
rm -rf build
rm -rf dist
rm -rf lake.egg-info

# 切换到python3，以便包通用
# source activate py35

# 本地打包，--universal适配python2与python3
python setup.py bdist_wheel --universal

# 上传
# 上传的账号写在~/.pypirc，形式如下:
# [pypi]
# username = <username>
# password = <password>
twine upload dist/*

# 退出python3，进入python2中
# source deactivate