import os
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append(now_dir + "/GPT_SoVITS")

import soundfile
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np
import librosa
import re
from text import cleaned_text_to_sequence
from text.chinese2 import g2p, text_normalize
import time
import onnxruntime as ort

splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }
def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text

def get_spepc(hps, filename):
    audio,sr = librosa.load(filename, sr=int(hps.sampling_rate))
    MAX_WAV_LEN = 320000
    if audio.shape[-1] < MAX_WAV_LEN:
        audio = np.pad(audio, (0, MAX_WAV_LEN - audio.shape[-1]), mode="constant")

    audio = audio.astype(np.float32)
    maxx = np.abs(audio).max()
    if maxx > 1:
        audio /= min(2, maxx)
    audio_norm = audio
    audio_norm = np.expand_dims(audio_norm, axis=0)
    spec = spectrogram_numpy(
        audio_norm,
        hps.filter_length,
        hps.hop_length,
        hps.win_length,
    )
    return spec

def spectrogram_numpy(y, n_fft, hop_size, win_size):
    if np.min(y) < -1.0:
        print("min value is ", np.min(y))
    if np.max(y) > 1.0:
        print("max value is ", np.max(y))

    fft_window = np.hanning(win_size).astype(np.float32)

    pad_len = int((n_fft - hop_size) / 2)
    y = np.pad(y, ((0, 0), (pad_len, pad_len)), mode="reflect")
    
    n_frames = 1 + (y.shape[-1] - n_fft) // hop_size
    frames = np.lib.stride_tricks.as_strided(y,
                                             shape=(n_frames, n_fft),
                                             strides=(y.strides[-1] * hop_size, y.strides[-1]))
    
    frames = frames * fft_window

    spec = np.fft.rfft(frames, n=n_fft)

    spec = np.abs(spec)
    spec = np.sqrt(spec**2 + 1e-6)

    spec = np.swapaxes(spec.astype(np.float32), 0, 1)
    spec = np.expand_dims(spec, axis=0)
    return spec


class HPS:
    sampling_rate = 32000  # 采样率
    filter_length = 2048   # FFT 窗口大小
    hop_length = 640       # 帧移
    win_length = 2048      # 窗长


class SSLModel:
    def __init__(self,cnhubert_base_path):
        self.model = ort.InferenceSession(cnhubert_base_path)
    
    def prepare_input(self, ref_wav_path):
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
            print("音频长度不符合要求")
            raise ValueError   
        zero_wav = np.zeros(int(32000 * 0.3),dtype=np.float32)
        wav16k = np.concatenate([wav16k, zero_wav], axis=0)
        wav16k = np.expand_dims(wav16k, axis=0)
        return wav16k, zero_wav

    def __call__(self, ref_audio_16k):
        ssl_content = self.model.run(None, {"ref_audio_16k": ref_audio_16k})[0]
        return ssl_content


class VitsEncoder:
    def __init__(self, vits_encoder_path):
        self.model = ort.InferenceSession(vits_encoder_path)
    
    def __call__(self, ssl_content):
        prompt = self.model.run(None, {"ssl_content": ssl_content})[0]
        return prompt
    
    
class BertModel:
    def __init__(self,bert_path):
        self.model = ort.InferenceSession(bert_path)

    def __call__(self, input_ids, token_type_ids, attention_mask):
        res = self.model.run(None, {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask})[0]  
        return res

    
class G2PWBertModel:
    def __init__(self, g2pw_path, bert_path):
        self.tokenizer = AutoTokenizer.from_pretrained(g2pw_path)
        self.bert_model = BertModel(bert_path)
    
    def prepare_input(self, text, prompt_text):
        prompt_text = prompt_text.strip("\n")
        if (prompt_text[-1] not in splits): prompt_text += "。" 
        print("实际输入的参考文本:", prompt_text)
        text = text.strip("\n")
        if (text[0] not in splits and len(get_first(text)) < 4): text = "。" + text
        print("实际输入的目标文本:", text)
        norm_text = text_normalize(text)
        norm_prompt_text = text_normalize(prompt_text)
        return norm_text, norm_prompt_text

    def get_bert(self, norm_text, word2ph):
        inputs = self.tokenizer(norm_text,return_tensors="np")
        a, b, c = inputs["input_ids"].astype(np.int64), inputs["token_type_ids"].astype(np.int64), inputs["attention_mask"].astype(np.int64)
        res = self.bert_model(a, b, c)

        assert len(word2ph) == len(norm_text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = np.tile(res[i], (word2ph[i], 1))
            phone_level_feature.append(repeat_feature)
        phone_level_feature = np.concatenate(phone_level_feature)
        bert_onnx = phone_level_feature.T
        return bert_onnx
    
    def __call__(self, text, prompt_text):
        norm_text, norm_prompt_text = self.prepare_input(text, prompt_text)
        print("处理后的参考文本:", norm_prompt_text, len(norm_prompt_text))
        print("处理后的目标文本:", norm_text, len(norm_text))

        phones1, word2ph1 = g2p(norm_text) #g2pw模型
        phones1 = cleaned_text_to_sequence(phones1, "v2")
        bert1 = self.get_bert(norm_text, word2ph1) #BERT模型

        phones2, word2ph2 = g2p(norm_prompt_text) #g2pw模型
        phones2 = cleaned_text_to_sequence(phones2, "v2")
        bert2 = self.get_bert(norm_prompt_text, word2ph2) #BERT模型

        bert = np.expand_dims(np.concatenate([bert2, bert1], 1), 0)

        all_phoneme_ids = phones2 + phones1
        all_phoneme_ids = np.expand_dims(np.int64(all_phoneme_ids), 0)
        return bert, all_phoneme_ids, phones1


class T2SModel:
    def __init__(self, encoder_path, first_stage_decoder, stage_decoder):
        self.encoder = ort.InferenceSession(encoder_path)
        self.first_stage_decoder = ort.InferenceSession(first_stage_decoder)
        self.stage_decoder = ort.InferenceSession(stage_decoder)

    def __call__(self, phoneme_ids, bert, prompts):
        x = self.encoder.run(None, {"phoneme_ids": phoneme_ids, "bert": bert})[0]
        y, k, v, y_emb, x_example = self.first_stage_decoder.run(None, {"x": x, "prompts": prompts})

        stop = False
        for idx in tqdm(range(1, 300)):

            y, k, v, y_emb, logits, samples = self.stage_decoder.run(None, {"iy": y, "ik": k, "iv": v, "iy_emb": y_emb, "x_example": x_example})

            if np.argmax(logits, axis=-1)[0] == 1024 or samples[0, 0] == 1024:
                stop = True
            if stop: break

        return y[...,-idx:]


class VitsDecoder:
    def __init__(self, vits_decoder_path):
        self.model = ort.InferenceSession(vits_decoder_path)
    
    def __call__(self, pred_semantic, text_seq, refer):
        return self.model.run(None, {"pred_semantic": pred_semantic, "text_seq": text_seq, "refer": refer})[0]


class GptSovits:
    def __init__(self, args):
        self.ssl_model = SSLModel(args.ssl_path)
        self.vits_encoder = VitsEncoder(args.vits_encoder_path)
        self.bert_model = G2PWBertModel(args.g2pw_path, args.bert_path)
        self.t2s_model = T2SModel(args.t2s_encoder_path, args.t2s_first_stage_decoder, args.t2s_stage_decoder)
        self.vits_decoder = VitsDecoder(args.vits_decoder_path)
        self.hps = HPS()
    
    def __call__(self, args):
        ref_wav_16k, zero_wav = self.ssl_model.prepare_input(args.ref_wav_path)
        ssl_content = self.ssl_model(ref_wav_16k)
        prompt = self.vits_encoder(ssl_content)
        bert, all_phoneme_ids, phones1 = self.bert_model(args.text, args.prompt_text)
        pred_semantic = self.t2s_model(all_phoneme_ids, bert, prompt)

        pred_semantic = np.expand_dims(pred_semantic, 0)
        phones1 = np.expand_dims(np.int64(phones1), 0)
        refer = get_spepc(self.hps, args.ref_wav_path)

        print("pred_semantic:", pred_semantic.shape, "phones1:",phones1.shape, "refer:", refer.shape)
        audio = self.vits_decoder(pred_semantic, phones1, refer)

        audio = audio[0,0]

        max_audio=np.abs(audio).max()
        if max_audio>1:audio/=max_audio

        audio_opt = []
        audio_opt.append(audio)
        audio_opt.append(zero_wav)
        audio = (np.concatenate(audio_opt, 0) * 32768).astype(np.int16)

        soundfile.write("out_test.wav", audio, self.hps.sampling_rate)

        return audio
        



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssl_path", type=str, default="models/dynamic/00_cnhubert.onnx")
    parser.add_argument("--vits_encoder_path", type=str, default="models/dynamic/01_vits_encoder.onnx")
    parser.add_argument("--g2pw_path", type=str, default="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large")
    parser.add_argument("--bert_path", type=str, default="models/dynamic/02_bert.onnx")
    parser.add_argument("--t2s_encoder_path", type=str, default="models/dynamic/03_t2s_encoder.onnx")
    parser.add_argument("--t2s_first_stage_decoder", type=str, default="models/dynamic/04_t2s_first_stage_decoder.onnx")
    parser.add_argument("--t2s_stage_decoder", type=str, default="models/dynamic/05_t2s_stage_decoder.onnx")
    parser.add_argument("--vits_decoder_path", type=str, default="models/dynamic/06_vits_decoder.onnx")

    parser.add_argument("--ref_wav_path", type=str, default="参考音频/说话-正是像停云小姐这样的接渡使往来周旋，仙舟的贸易才能有如今的繁盛。.wav")
    parser.add_argument("--text", type=str, default="大家好，我是来自四川理塘的王真先生。") #三个标点符号。首句少于四个字，两个标点符号。
    parser.add_argument("--prompt_text", type=str, default="。正是像停云小姐这样的接渡使往来周旋，仙舟的贸易才能有如今的繁盛。") #三个标点符号

    args = parser.parse_args()
    start = time.time()

    gptsovits = GptSovits(args)
    gptsovits(args)

    print("put time : ",time.time() - start) # 20s



