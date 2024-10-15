# GPT-SoVITS-TPU

- [x] Test on 1684x server (tpu_dev docker with Miniconda) -- PASS
- [x] Test on SG2300X -- PASS
  
## Prepare the environment

Please use python version 3.10. The system python version of SG2300X is 3.8, you can use pyenv to install version 3.10 for this project. The installation process can be referenced: [https://www.cnblogs.com/safe-rabbit/p/17130336.html](https://www.cnblogs.com/safe-rabbit/p/17130336.html).

### Install dependencies

```bash
cd GPT-SoVITS-TPU
pip install -r requirements.txt
pip install python_wheels/sophon_arm-3.8.0-py3-none-any.whl
# pip install python_wheels/tpu_perf-1.2.60-py3-none-manylinux2014_x86_64.whl
```

### Download by Git

```
git clone https://www.modelscope.cn/wlc952/GPT-SoVITS-TPU.git
```

## Inference on SG2300X

It takes about 20 seconds to generate a speech of about 30 words on SG2300X.

```bash
python run_by_sail.py
```

or use webui

```bash
python webui_sail.py
```