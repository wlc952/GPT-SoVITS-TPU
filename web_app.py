import gradio as gr
import numpy as np
import time
import logging
from main import *
from utils import *

# 设置日志记录
logging.basicConfig(level=logging.INFO)

class GptSovits_long(GptSovits):
    def __init__(self):
        super().__init__()

    def prepare_input(self, text):
        text = text.strip("\n")
        if text[-1] not in splits:
            text += "。"
        norm_text = text_normalize(text)
        return norm_text

    def get_bert(self, norm_text, word2ph, len_ori_text):
        inputs = self.tokenizer(norm_text, return_tensors="np")
        a, b, c = inputs["input_ids"].astype(np.int32), inputs["token_type_ids"].astype(np.int32), inputs["attention_mask"].astype(np.int32)
        c[..., len_ori_text:] = 0

        res = self.bmodels([a, b, c], net_name='02_bert_35')[0]

        assert len(word2ph) == len(norm_text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = np.tile(res[i], (word2ph[i], 1))
            phone_level_feature.append(repeat_feature)
        phone_level_feature = np.concatenate(phone_level_feature)
        bert = phone_level_feature.T
        return bert

    def g2pw_bert_process(self, text, type='target'):
        norm_text = self.prepare_input(text)
        len_ori_text = len(norm_text)

        if type == 'target':
            num_dots = 3
        else:
            num_dots = 2
        how_many = sum(1 for x in norm_text if x in splits)
        delta = num_dots - how_many
        if delta > 0:
            norm_text = "." * delta + norm_text


        norm_text = fix_text_lenth(norm_text, fix_len=35)

        logging.info(f"文本处理完成：“{norm_text}”, 长度：{len(norm_text)}")

        phones1, word2ph1 = g2p(norm_text)  # g2pw模型
        phones1 = cleaned_text_to_sequence(phones1, "v2")
        id = find_penultimate_tag(phones1)
        phones1[id + 1:] = [2] * (len(phones1) - id - 1)

        bert1 = self.get_bert(norm_text, word2ph1, len_ori_text)  # BERT模型

        return bert1, phones1

    def __call__(self, ref_wav_path, ref_text, target_text, top_k=15, top_p=1.0, post_process=True, min_silence_len=200):
        try:
            start = time.time()
            ref_wav_16k = prepare_wav_input(ref_wav_path)
            ssl_content = self.bmodels([ref_wav_16k], net_name='00_cnhubert')[0]
            prompt = self.bmodels([ssl_content], net_name='01_vits_encoder')[0]

            bert2, phones2 = self.g2pw_bert_process(ref_text, fix_len=35)

            texts = process_long_text(target_text)
            audio_list = []

            for text in texts:
                if text[-1] not in ['。', '！', '？']:
                    text = text[:-1] + '。'
                bert1, phones1 = self.g2pw_bert_process(text, fix_len=85)

                bert = np.expand_dims(np.concatenate([bert2, bert1], 1), 0)
                all_phoneme_ids = phones2 + phones1
                all_phoneme_ids = np.expand_dims(np.int32(all_phoneme_ids), 0)

                pred_semantic = self.t2s_process(all_phoneme_ids, bert, prompt, top_k, top_p)

                SET_PRED_SEMANTIC_LEN = 600
                if pred_semantic.shape[-1] < SET_PRED_SEMANTIC_LEN:
                    padding_size = SET_PRED_SEMANTIC_LEN - pred_semantic.shape[-1]
                    pred_semantic = np.pad(pred_semantic, pad_width=((0, 0), (0, padding_size)), mode='edge')

                pred_semantic = np.expand_dims(pred_semantic, 0)
                phones1 = np.expand_dims(np.int32(phones1), 0)
                refer = get_spepc(self.hps, ref_wav_path)

                audio_np = self.bmodels([pred_semantic, phones1, refer, self.randn_np], net_num=-1)[0]
                audio_np = audio_np[0, 0]

                max_audio = np.abs(audio_np).max()
                if max_audio > 1:
                    audio_np /= max_audio

                audio_test = (audio_np * 32768).astype(np.int16)
                if post_process:
                    sr, audio_test = audio_post_process(self.hps.sampling_rate, audio_test, min_silence_len) 
                audio_list.append(audio_test)

            audio_test = np.concatenate(audio_list)

            logging.info(f"耗时：{time.time() - start}")
            return self.hps.sampling_rate, audio_test
        except Exception as e:
            logging.error(f"处理音频时出错: {e}")
            raise

gptsovits = GptSovits_long()

def process_audio(audio, ref_text, target_text, top_k, top_p, post_process=True, min_silence_len=200):
    sr, audio = gptsovits(audio, ref_text, target_text, top_k=top_k, top_p=top_p, post_process=post_process, min_silence_len=min_silence_len)
    return sr, audio

iface = gr.Interface(
    fn=process_audio,
    inputs=[
        gr.Audio(sources="upload", type="filepath", label="参考语音(5~10s)"),
        gr.Textbox(lines=2, placeholder="输入参考文本...", label="参考文本(三个标点，35字以内。)"),
        gr.Textbox(lines=2, placeholder="输入目标文本...", label="目标文本(四句一切，最好标点为4N个。)"),
        gr.Slider(value=15, label="Top-k", minimum=1, maximum=100, step=1),
        gr.Slider(value=1.0, label="Top-p", minimum=0.5, maximum=1.0, step=0.01),
        gr.Checkbox(value=True, label="是否进行后处理"),
        gr.Slider(value=200, label="静音检测阈值(毫秒)", minimum=100, maximum=500, step=50),
    ],
    outputs=gr.Audio(label="生成的语音"),
    title="GPT-SoVITS音色克隆",
    description="上传一段参考语音，并输入参考文本和目标文本，将生成一段新的语音。",
)

iface.launch()

