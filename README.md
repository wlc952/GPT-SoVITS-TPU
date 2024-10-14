# GPT-SoVITS-TPU

**New version that supports up to 85 characters and is a simplified version of the project, address: https://www.modelscope.cn/models/wlc952/GPT-SoVITS-TPU**
![image](https://github.com/user-attachments/assets/72c7a966-9299-402b-9523-603985a3e50b)

  
## Prepare the environment

Please use python version 3.10. The system python version of SG2300X is 3.8, you can use pyenv to install version 3.10 for this project. The installation process can be referenced: [https://www.cnblogs.com/safe-rabbit/p/17130336.html](https://www.cnblogs.com/safe-rabbit/p/17130336.html).

### Download this project

```bash
git clone https://github.com/wlc952/GPT-SoVITS-TPU.git
```

### Install dependencies

```bash
cd GPT-SoVITS-TPU
pip install -r requirements.txt
pip install sophon_arm-3.8.0-py3-none-any.whl
```

### Dowload bmodels

Download models from <https://huggingface.co/wlc952/GPT-SoVITS-TPU/tree/main>

The models folder structure is as follows:

```bash
.
+--- 00_cnhubert_1684x_f32.bmodel
+--- 01_vits_encoder_1684x_f32.bmodel
+--- 02_bert_1684x_f32.bmodel
+--- 03_t2s_encoder_1684x_f32.bmodel
+--- 04_t2s_embedding_1684x_f32.bmodel
+--- 05_t2s_attnmask.onnx
+--- 06_t2s_first_step_decoder_1684x_f32.bmodel
+--- 07_t2s_ar_predict_1684x_f32.bmodel
+--- 08_t2s_sample_layer.onnx
+--- 09_t2s_update_next_step_1684x_f32.bmodel
+--- 10_t2s_next_step_decoder_1684x_f32.bmodel
+--- 11_vits_decoder_1684x_f32.bmodel
```

## Inference on SG2300X

It takes about 20 seconds to generate a speech of about 30 words on SG2300X.

```bash
python infer_bmodel_cache.py
```
