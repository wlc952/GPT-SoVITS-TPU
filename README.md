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

+--- dynamic
|   +--- 00_cnhubert.onnx
|   +--- 01_vits_encoder.onnx
|   +--- 02_bert.onnx
|   +--- 03_t2s_encoder.onnx
|   +--- 04_t2s_first_stage_decoder.onnx
|   +--- 05_t2s_stage_decoder.onnx
|   +--- 06_vits_decoder.onnx
+--- g2pw_tokenizer
|   +--- config.json
|   +--- tokenizer.json
```

## Inference on TPU

```bash
python infer_bmodel_cache.py
```
