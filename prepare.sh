#!/bin/bash

# 安装 libsndfile1
sudo apt-get install libsndfile1 -y

# 升级 pip
pip install --upgrade pip
pip install python_wheels/sophon_arm-3.8.0-py38-none-any.whl

# 安装 requirements.txt 中的依赖
pip install -r requirements.txt