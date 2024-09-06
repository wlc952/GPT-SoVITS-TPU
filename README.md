# GPT-SoVITS-TPU

- [x] Test on 1684x server (tpu_dev docker with Miniconda) -- PASS
- [ ] Test on Airbox -- TODO

## Dowload bmodels

Download models from <https://huggingface.co/wlc952/GPT-SoVITS-TPU/tree/main>

The models folder structure is as follows:

```bash
+--- bmodel
|   +--- 00_cnhubert_1684x_f32.bmodel
|   +--- 01_vits_encoder_1684x_f32.bmodel
|   +--- 02_bert_1684x_f32.bmodel
|   +--- 03_t2s_encoder_1684x_f32.bmodel
|   +--- 04_embedding_1684x_f32.bmodel
|   +--- 05_attnmask.onnx
|   +--- 06_t2s_decoder_1684x_f32.bmodel
|   +--- 07_ar_predict_1684x_f32.bmodel
|   +--- 08_sample.onnx
|   +--- 09_vits_decoder_1684x_f32.bmodel
+--- cache
|   +--- ar_predict_1684x_f32.bmodel
|   +--- blocks
|   |   +--- block_0.bmodel
|   |   +--- block_1.bmodel
|   |   +--- block_10.bmodel
|   |   +--- block_11.bmodel
|   |   +--- block_12.bmodel
|   |   +--- block_13.bmodel
|   |   +--- block_14.bmodel
|   |   +--- block_15.bmodel
|   |   +--- block_16.bmodel
|   |   +--- block_17.bmodel
|   |   +--- block_18.bmodel
|   |   +--- block_19.bmodel
|   |   +--- block_2.bmodel
|   |   +--- block_20.bmodel
|   |   +--- block_21.bmodel
|   |   +--- block_22.bmodel
|   |   +--- block_23.bmodel
|   |   +--- block_3.bmodel
|   |   +--- block_4.bmodel
|   |   +--- block_5.bmodel
|   |   +--- block_6.bmodel
|   |   +--- block_7.bmodel
|   |   +--- block_8.bmodel
|   |   +--- block_9.bmodel
|   +--- embedding_1684x_f32.bmodel
|   +--- first_stage_decoder_1684x_f32.bmodel
|   +--- mask.onnx
|   +--- sample_layer.onnx
|   +--- t2s_transformer.onnx
|   +--- update_next_step_1684x_f32.bmodel
+--- g2pw_tokenizer
|   +--- config.json
|   +--- tokenizer.json
```

## Inference on TPU

It takes about 16 seconds to generate a speech of about 30 words.

```bash
python infer_bmodel_cache.py
```
