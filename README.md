---
frameworks:
- other
license: MIT License
tasks:
- text-to-speech

#model-type:
##如 gpt、phi、llama、chatglm、baichuan 等
#- gpt

#domain:
##如 nlp、cv、audio、multi-modal
#- nlp

#language:
##语言代码列表 https://help.aliyun.com/document_detail/215387.html?spm=a2c4g.11186623.0.0.9f8d7467kni6Aa
#- cn 

#metrics:
##如 CIDEr、Blue、ROUGE 等
#- CIDEr

#tags:
##各种自定义，包括 pretrained、fine-tuned、instruction-tuned、RL-tuned 等训练方法和其他
#- pretrained

#tools:
##如 vllm、fastchat、llamacpp、AdaSeq 等
#- vllm
---

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