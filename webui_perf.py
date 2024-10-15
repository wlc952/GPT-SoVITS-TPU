import gradio as gr
import numpy as np
import time
from run_by_perf import *


class GptSovits:
    def __init__(self,):
        model_path = "models"
        self.ssl_model = BModel(model_path + "/00_cnhubert.bmodel")
        self.vits_encoder = BModel(model_path + "/01_vits_encoder.bmodel")
        self.bert_model = G2PWBertModel(model_path)
        self.t2s_model = T2SModel(model_path)
        self.vits_decoder = BModel(model_path + "/11_vits_decoder.bmodel")
        self.hps = HPS()
        self.randn_np = np.random.randn(1, 192, 1200).astype(np.float32)
    
    def __call__(self, audio_path, ref_text, target_text):
        start = time.time()
        ref_wav_16k= prepare_wav_input(audio_path)
        ssl_content = self.ssl_model([ref_wav_16k])[0]
        prompt = self.vits_encoder([ssl_content])[0]
        bert, all_phoneme_ids, phones1 = self.bert_model(target_text, ref_text)

        pred_semantic = self.t2s_model(all_phoneme_ids, bert, prompt)

        SET_PRED_SEMANTIC_LEN = 600
        if pred_semantic.shape[-1] < SET_PRED_SEMANTIC_LEN:
            padding_size = SET_PRED_SEMANTIC_LEN - pred_semantic.shape[-1]
            pred_semantic = np.pad(pred_semantic, pad_width=((0, 0), (0, padding_size)), mode='edge')

        pred_semantic = np.expand_dims(pred_semantic, 0)
        phones1 = np.expand_dims(np.int32(phones1), 0)
        refer = get_spepc(self.hps, audio_path)
        
        audio_np = self.vits_decoder([pred_semantic, phones1, refer, self.randn_np])[0]

        audio_np = audio_np[0,0]
        max_audio=np.abs(audio_np).max()
        if max_audio>1: audio_np/=max_audio

        audio_test = (audio_np * 32768).astype(np.int16)
        # soundfile.write("out_test.wav", audio_test, self.hps.sampling_rate)

        print("耗时：", time.time() - start)
        return self.hps.sampling_rate, audio_test

gptsovits = GptSovits()

def process_audio(audio, ref_text, target_text):
    return gptsovits(audio, ref_text, target_text)

iface = gr.Interface(
    fn=process_audio,
    inputs=[
        gr.Audio(sources="upload", type="filepath", label="参考语音(5~10s)"),
        gr.Textbox(lines=2, placeholder="输入参考文本...", label="参考文本(三个标点，35字以内。)"),
        gr.Textbox(lines=2, placeholder="输入目标文本...", label="目标文本(四个标点，85字以内。)")
    ],
    outputs=gr.Audio(label="生成的语音"),
    title="GPT-SoVITS语音生成器",
    description="上传一段参考语音，并输入参考文本和目标文本，生成一段新的语音。",
)

iface.launch()