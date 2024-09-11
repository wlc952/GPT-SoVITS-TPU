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
from GPT_SoVITS.text import cleaned_text_to_sequence
from GPT_SoVITS.text.chinese2 import g2p, text_normalize
import time
import onnxruntime as ort
from tpu_perf.infer import SGInfer
import sophon.sail as sail

class BModel:
    def __init__(self, model_path="", output_names="", device_id=0):
        if "DEVICE_ID" in os.environ:
            device_id = int(os.environ["DEVICE_ID"])
            # print(">>>> device_id is in os.environ. and device_id = ", device_id)
        self.model_path = model_path
        self.device_id = device_id
        try:
            self.model = sail.Engine(model_path, device_id, sail.IOMode.SYSIO)
        except Exception as e:
            print("load model error; please check model path and device status;")
            print(">>>> model_path: ", model_path)
            print(">>>> device_id: ", device_id)
            print(">>>> sail.Engine error: ", e)
            raise e
        self.graph_name = self.model.get_graph_names()[0]
        self.input_name = self.model.get_input_names(self.graph_name)
        self.output_name = self.model.get_output_names(self.graph_name)

    def __str__(self):
        return "EngineOV: model_path={}, device_id={}".format(self.model_path, self.device_id)

    def __call__(self, args):
        if isinstance(args, list):
            values = args
        elif isinstance(args, dict):
            values = list(args.values())
        else:
            raise TypeError("args is not list or dict")
        args = {}
        for i in range(len(values)):
            args[self.input_name[i]] = values[i]
        output = self.model.process(self.graph_name, args)
        return list(output.values())

# class BModel:
#     def __init__(self, model_path="", batch=1, device_id=(0,)) :
#         self.model_path = model_path
#         self.model = SGInfer(model_path , batch=batch, devices=device_id)
#         self.device_id = device_id
        
#     def __str__(self):
#         return "EngineOV: model_path={}, device_id={}".format(self.model_path,self.device_id)
        
#     def __call__(self, args):
#         if isinstance(args, list):
#             values = args
#         elif isinstance(args, dict):
#             values = list(args.values())
#         else:
#             raise TypeError("args is not list or dict")
#         task_id = self.model.put(*values)
#         task_id, results, valid = self.model.get()
#         return results



splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }

def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text

def find_penultimate_tag(lst):
    tag = [0,3,4,321]
    count = 0
    for i in range(len(lst)-1, -1, -1):
        if lst[i] in tag:
            count += 1
            if count == 2:
                return i
    return -1

def fix_text_lenth(text, max_len = 35):
    if len(text) > max_len:
        print(f"文本长度超过{max_len}个字符")
        raise ValueError
    elif len(text) == max_len:
        pass
    else:
        text = text + '零' * (max_len - len(text) - 1) + '.'
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

def prepare_wav_input(ref_wav_path):
    wav16k, sr = librosa.load(ref_wav_path, sr=16000)
    if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
        print("音频长度不符合要求")
        raise ValueError

    fixed_length = 160000
    # 如果语音太短，则填充零
    if wav16k.shape[0] < fixed_length:
        padding = fixed_length - wav16k.shape[0]
        wav16k = np.pad(wav16k, (0, padding), 'constant')

    wav16k = np.expand_dims(wav16k, axis=0)
    return wav16k


class HPS:
    sampling_rate = 32000  # 采样率
    filter_length = 2048   # FFT 窗口大小
    hop_length = 640       # 帧移
    win_length = 2048      # 窗长





    
class G2PWBertModel:
    def __init__(self, g2pw_path, bert_path):
        self.tokenizer = AutoTokenizer.from_pretrained(g2pw_path)
        self.bert_model = BModel(bert_path)
    
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

    def get_bert(self, norm_text, word2ph, len_ori_text):
        inputs = self.tokenizer(norm_text,return_tensors="np")
        a, b, c = inputs["input_ids"].astype(np.int32), inputs["token_type_ids"].astype(np.int32), inputs["attention_mask"].astype(np.int32)
        c[...,len_ori_text:] = 0

        res = self.bert_model([a, b, c])[0]

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

        len_ori_text = len(norm_text)
        len_ori_prompt_text = len(norm_prompt_text)

        norm_prompt_text = fix_text_lenth(norm_prompt_text)
        norm_text = fix_text_lenth(norm_text)

        print("处理后的参考文本:", norm_prompt_text, len(norm_prompt_text))
        print("处理后的目标文本:", norm_text, len(norm_text))

        phones1, word2ph1 = g2p(norm_text) #g2pw模型
        phones1 = cleaned_text_to_sequence(phones1, "v2")
        len_phones = len(phones1)
        id = find_penultimate_tag(phones1)
        phones1[id+1:] = [2] * (len(phones1) - id - 1)# "_":95, "-":2, '停': 351, '空':352

        bert1 = self.get_bert(norm_text, word2ph1, len_ori_text) #BERT模型

        phones2, word2ph2 = g2p(norm_prompt_text) #g2pw模型
        phones2 = cleaned_text_to_sequence(phones2, "v2")
        len_phones += len(phones2)
        id = find_penultimate_tag(phones2)
        phones2[id+1:] = [2] * (len(phones2) - id - 1)

        bert2 = self.get_bert(norm_prompt_text, word2ph2, len_ori_prompt_text) #BERT模型

        bert = np.expand_dims(np.concatenate([bert2, bert1], 1), 0)

        all_phoneme_ids = phones2 + phones1
        all_phoneme_ids = np.expand_dims(np.int32(all_phoneme_ids), 0)
        return bert, all_phoneme_ids, phones1


class T2SStage:
    def __init__(self, dir):
        self.blocks = []
        for i in range(24):
            self.blocks.append(BModel(dir + f"blocks/block_{i}.bmodel"))
    
    def __call__(self, x, k_cache, v_cache, current_size):
        zero_np = np.zeros((1, 1023 - current_size, 512), dtype=np.float32)

        k_list = []
        v_list = []

        for i in range(24):
            ik, iv, x = self.blocks[i]([x, k_cache[i], v_cache[i], np.int32([current_size])])
            ik = np.concatenate([ik, zero_np], axis=1)
            iv = np.concatenate([iv, zero_np], axis=1)
            k_list.append(ik)
            v_list.append(iv)
        return x, k_list, v_list, current_size + 1
        
        
class T2SModel:
    def __init__(self, encoder_path, cache_dir):
        self.encoder = BModel(encoder_path)
        self.embedding = BModel(cache_dir + 'embedding_1684x_f32.bmodel')
        self.attnmask = ort.InferenceSession(cache_dir + 'mask.onnx')
        self.first_stage_decoder = BModel(cache_dir + 'first_stage_decoder_1684x_f32.bmodel')
        self.ar_predict_layer = BModel(cache_dir + 'ar_predict_1684x_f32.bmodel')
        self.sample_layer = ort.InferenceSession(cache_dir + 'sample_layer.onnx')
        self.update_next_step = BModel(cache_dir + 'update_next_step_1684x_f32.bmodel')
        self.decoder = T2SStage(cache_dir)
        # self.decoder = ort.InferenceSession(cache_dir + 't2s_transformer.onnx')

    def __call__(self, phoneme_ids, bert, prompts):
        x = self.encoder([phoneme_ids, bert])[0]
        y = prompts

        y_len = y.shape[1]
        x_len = x.shape[1]

        xy_pos = self.embedding([x, y])[0]
        xy_attn_mask = self.attnmask.run(None, {"x_len": np.int64([x_len]), "y_len": np.int64([y_len])})[0]
        k_cache, v_cache, xy_dec = self.first_stage_decoder([xy_pos, xy_attn_mask])
        logits = self.ar_predict_layer([xy_dec[:, -1]])[0]
        logits = logits[:, :-1]   
        samples, y = self.sample_layer.run(None, {"logits": logits[0], "y": np.int64(y)})

        y = np.int32(y)
        xy_pos = self.update_next_step([y[:, -1:], np.int32([y_len])])[0]

        current_size = 381
        for idx in tqdm(range(1,500)):
            ########### method 1 : self.decoder use 'models/cache/blocks/*.bmodel' ###############
            xy_dec, k_cache, v_cache, current_size = self.decoder(xy_pos, k_cache, v_cache, current_size)

            ########### method 2 : self.decoder use 'models/cache/t2s_transformer.onnx' ###############
            # xy_dec, k, v = self.decoder.run(None, {"xy_pos":xy_pos,"ik":k_cache, "iv":v_cache, "current_size":np.int64([current_size])})
            # k_cache[:, :, current_size: current_size+1, :] = k
            # v_cache[:, :, current_size: current_size+1, :] = v
            # current_size = current_size + 1

            logits = self.ar_predict_layer([xy_dec[:, -1]])[0]
            samples, y = self.sample_layer.run(None, {"logits": logits[0], "y": np.int64(y)})
            y = np.int32(y)
            
            if np.argmax(logits, axis=-1)[0] == 1024 or samples[0, 0] == 1024:  break

            xy_pos = self.update_next_step([y[:, -1:], np.int32([idx + y_len])])[0]
                    
        y = y[:, :-1]
        idx = idx -1
        return y[...,-idx:]



class GptSovits:
    def __init__(self, args):
        self.ssl_model = BModel(args.ssl_path)
        self.vits_encoder = BModel(args.vits_encoder_path)
        self.bert_model = G2PWBertModel(args.g2pw_path, args.bert_path)
        self.t2s_model = T2SModel(args.t2s_encoder_path, args.t2s_cache_decoder_path)
        self.vits_decoder = BModel(args.vits_decoder_path)
        self.hps = HPS()
    
    def __call__(self, args):
        start = time.time()

        ref_wav_16k= prepare_wav_input(args.ref_wav_path)
        ssl_content = self.ssl_model([ref_wav_16k])[0]
        prompt = self.vits_encoder([ssl_content])[0]
        bert, all_phoneme_ids, phones1 = self.bert_model(args.text, args.prompt_text)
        pred_semantic = self.t2s_model(all_phoneme_ids, bert, prompt)

        SET_PRED_SEMANTIC_LEN = 300
        if pred_semantic.shape[-1] < SET_PRED_SEMANTIC_LEN:
            padding_size = SET_PRED_SEMANTIC_LEN - pred_semantic.shape[-1]
            pred_semantic = np.pad(pred_semantic, pad_width=((0, 0), (0, padding_size)), mode='edge')

        pred_semantic = np.expand_dims(pred_semantic, 0)
        phones1 = np.expand_dims(np.int32(phones1), 0)
        refer = get_spepc(self.hps, args.ref_wav_path)
        randn_np = np.random.randn(1, 192, 600).astype(np.float32)

        print("pred_semantic:", pred_semantic.shape, "phones1:",phones1.shape, "refer:", refer.shape)
        audio_np = self.vits_decoder([pred_semantic, phones1, refer, randn_np])[0]

        audio_np = audio_np[0,0]
        audio_np = audio_np[:32000*5]
        max_audio=np.abs(audio_np).max()
        if max_audio>1: audio_np/=max_audio

        audio_test = (audio_np * 32768).astype(np.int16)
        soundfile.write("out_test.wav", audio_test, self.hps.sampling_rate)

        print("耗时：", time.time() - start)
        return audio_test
        



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssl_path", type=str, default="models/bmodel/00_cnhubert_1684x_f32.bmodel")
    parser.add_argument("--vits_encoder_path", type=str, default="models/bmodel/01_vits_encoder_1684x_f32.bmodel")
    parser.add_argument("--g2pw_path", type=str, default="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large")
    parser.add_argument("--bert_path", type=str, default="models/bmodel/02_bert_1684x_f32.bmodel")
    parser.add_argument("--t2s_encoder_path", type=str, default="models/bmodel/03_t2s_encoder_1684x_f32.bmodel")
    parser.add_argument("--t2s_cache_decoder_path", type=str, default="models/cache/")
    parser.add_argument("--vits_decoder_path", type=str, default="models/bmodel/09_vits_decoder_1684x_f32.bmodel")

    parser.add_argument("--ref_wav_path", type=str, default="参考音频/说话-正是像停云小姐这样的接渡使往来周旋，仙舟的贸易才能有如今的繁盛。.wav")
    parser.add_argument("--text", type=str, default="大家好，我是来自四川理塘的王真先生。") #三个标点符号。首句少于四个字，两个标点符号。
    parser.add_argument("--prompt_text", type=str, default="。正是像停云小姐这样的接渡使往来周旋，仙舟的贸易才能有如今的繁盛。") #三个标点符号

    args = parser.parse_args()
    start = time.time()

    a = GptSovits(args)
    a(args)
    a(args)
    a(args)
    print("总耗时：",time.time() - start)







