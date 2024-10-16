import os
import sys
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
import sophon.sail as sail

now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(now_dir)
sys.path.append(now_dir+'/text')


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

def fix_text_lenth(text, max_len=35):
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
    # 如果语音太长，则截断
    elif wav16k.shape[0] > fixed_length:
        wav16k = wav16k[:fixed_length]

    wav16k = np.expand_dims(wav16k, axis=0)
    return wav16k


class HPS:
    sampling_rate = 32000  # 采样率
    filter_length = 2048   # FFT 窗口大小
    hop_length = 640       # 帧移
    win_length = 2048      # 窗长


class BmodelLoader:
    def __init__(self, model_path, device_id=0) :
        self.model_path = model_path
        self.device_id = device_id
        try:
            self.model = sail.Engine(model_path, device_id, sail.IOMode.SYSIO)
        except Exception as e:
            print("load model error; please check model path and device status;")
            print(">>>> model_path: ",model_path)
            print(">>>> device_id: ",device_id)
            print(">>>> sail.Engine error: ",e)
            raise e
        sail.set_print_flag(False)
        self.graph_name = self.model.get_graph_names()
        self.input_name = []
        self.output_name = []
        for name in self.graph_name:
            self.input_name.append(self.model.get_input_names(name))
            self.output_name.append(self.model.get_output_names(name))

    def __str__(self):
        return "EngineOV: model_path={}, device_id={}".format(self.model_path,self.device_id)
    
    def __call__(self, args, net_name=None, net_num=0):
        if isinstance(args, list):
            values = args
        elif isinstance(args, dict):
            values = list(args.values())
        else:
            raise TypeError("args is not list or dict")
        args = {}
        if net_name is not None:
            graph_name = net_name
            input_name = self.model.get_input_names(net_name)
            output_name = self.model.get_output_names(net_name)
        elif net_num is not None:
            graph_name = self.graph_name[net_num]
            input_name = self.input_name[net_num]
            output_name = self.output_name[net_num]
        else:
            input_name = self.input_name[0]
            output_name = self.output_name[0]

        for i in range(len(values)):
            args[input_name[i]] = values[i]

        output = self.model.process(graph_name, args)
        res = []
        for name in output_name:
            res.append(output[name])
        return res


class GptSovits:
    def __init__(self, model_path="models", tokenizer="g2pw_tokenizer"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.bmodels = BmodelLoader(model_path + '/gptsovits.bmodel')
        self.attnmask = ort.InferenceSession(model_path + '/05_t2s_attnmask.onnx')
        self.sample_layer = ort.InferenceSession(model_path + '/08_t2s_sample_layer.onnx')

        self.hps = HPS()
        self.randn_np = np.random.randn(1, 192, 400).astype(np.float32)

    def prepare_input(self, text, prompt_text):
        prompt_text = prompt_text.strip("\n")
        if (prompt_text[-1] not in splits): prompt_text += "。" 
        # print("实际输入的参考文本:", prompt_text)
        text = text.strip("\n")
        # print("实际输入的目标文本:", text)
        norm_text = text_normalize(text)
        norm_prompt_text = text_normalize(prompt_text)
        return norm_text, norm_prompt_text

    def get_bert(self, norm_text, word2ph, len_ori_text):
        inputs = self.tokenizer(norm_text,return_tensors="np")
        a, b, c = inputs["input_ids"].astype(np.int32), inputs["token_type_ids"].astype(np.int32), inputs["attention_mask"].astype(np.int32)
        c[...,len_ori_text:] = 0


        res = self.bmodels([a, b, c], net_name='02_bert_35')[0]

        assert len(word2ph) == len(norm_text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = np.tile(res[i], (word2ph[i], 1))
            phone_level_feature.append(repeat_feature)
        phone_level_feature = np.concatenate(phone_level_feature)
        bert_onnx = phone_level_feature.T
        return bert_onnx
    
    def g2pw_bert_process(self, text, prompt_text):
        norm_text, norm_prompt_text = self.prepare_input(text, prompt_text)

        len_ori_text = len(norm_text)
        len_ori_prompt_text = len(norm_prompt_text)

        norm_prompt_text = fix_text_lenth(norm_prompt_text, 35)
        norm_text = fix_text_lenth(norm_text, 35)

        # print("处理后的参考文本:", norm_prompt_text, len(norm_prompt_text))
        # print("处理后的目标文本:", norm_text, len(norm_text))

        phones1, word2ph1 = g2p(norm_text) #g2pw模型
        phones1 = cleaned_text_to_sequence(phones1, "v2")
        len_phones = len(phones1)
        id = find_penultimate_tag(phones1)
        phones1[id+1:] = [2] * (len(phones1) - id - 1)# "_":95, "-":2

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
    
    def t2s_process(self, phoneme_ids, bert, prompts, top_k=15):
        sample_dict = {"top_k": np.int64([top_k])}
                           
        x = self.bmodels([phoneme_ids, bert], net_name='03_t2s_encoder')[0]
        y = prompts

        y_len = y.shape[1]
        x_len = x.shape[1]

        xy_pos = self.bmodels([x, y], net_name='04_t2s_embedding')[0]
        xy_attn_mask = self.attnmask.run(None, {"x_len": np.int64([x_len]), "y_len": np.int64([y_len])})[0]

        xy_dec, k_cache, v_cache = self.bmodels([xy_pos, xy_attn_mask], net_name='06_t2s_first_step_decoder')

        logits = self.bmodels([xy_dec[:, -1]], net_name='07_t2s_predict_layer')[0]
        logits = logits[:, :-1]

        sample_dict["logits"] = logits[0]
        sample_dict["y"] = np.int64(y)
        samples, y = self.sample_layer.run(None, sample_dict)

        y = np.int32(y)
        xy_pos = self.bmodels([y[:, -1:], np.int32([y_len])], net_name='09_t2s_update_next_step')[0]

        for idx in tqdm(range(1,301)):
            xy_dec, k_cache, v_cache = self.bmodels([xy_pos, k_cache, v_cache], net_name='10_t2s_next_step_decoder')

            logits = self.bmodels([xy_dec[:, -1]], net_name='07_t2s_predict_layer')[0]

            sample_dict["logits"] = logits[0]
            sample_dict["y"] = np.int64(y)
            samples, y = self.sample_layer.run(None, sample_dict)

            y = np.int32(y)
            
            if np.argmax(logits, axis=-1)[0] == 1024 or samples[0, 0] == 1024:  break

            xy_pos = self.bmodels([y[:, -1:], np.int32([idx + y_len])], net_name='09_t2s_update_next_step')[0]
                    
        y = y[:, :-1]
        idx = idx -1
        return y[...,-idx:]
    
    def __call__(self, ref_wav_path, text, prompt_text, top_k=15):
        start = time.time()
        ref_wav_16k= prepare_wav_input(ref_wav_path)
        ssl_content = self.bmodels([ref_wav_16k], net_name='00_cnhubert')[0]
        prompt = self.bmodels([ssl_content], net_name='01_vits_encoder')[0]
        bert, all_phoneme_ids, phones1 = self.g2pw_bert_process(text, prompt_text)

        pred_semantic = self.t2s_process(all_phoneme_ids, bert, prompt, top_k)

        SET_PRED_SEMANTIC_LEN = 200
        if pred_semantic.shape[-1] < SET_PRED_SEMANTIC_LEN:
            padding_size = SET_PRED_SEMANTIC_LEN - pred_semantic.shape[-1]
            pred_semantic = np.pad(pred_semantic, pad_width=((0, 0), (0, padding_size)), mode='edge')

        pred_semantic = np.expand_dims(pred_semantic, 0)
        phones1 = np.expand_dims(np.int32(phones1), 0)
        refer = get_spepc(self.hps, ref_wav_path)
        
        audio_np = self.bmodels([pred_semantic, phones1, refer, self.randn_np], net_num=-1)[0]
        audio_np = audio_np[0,0]

        max_audio=np.abs(audio_np).max()
        if max_audio>1: audio_np/=max_audio

        audio_test = (audio_np * 32768).astype(np.int16)
        soundfile.write("out_test.wav", audio_test, self.hps.sampling_rate)
        print("耗时：", time.time() - start)
        return self.hps.sampling_rate, audio_test
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models")
    parser.add_argument("--tokenizer", type=str, default="g2pw_tokenizer")

    parser.add_argument("--ref_wav_path", type=str, default="参考音频/说话-杨先生问的好问题，我一时半会儿也答不上来。容我想想…….wav")
    parser.add_argument("--text", type=str, default="内容还和芯产品硬件前端相呼应。正是像停云小姐这样的节度使往来周旋。") # 两个标点，35字以内。
    parser.add_argument("--prompt_text", type=str, default="杨先生问的好问题，我一时半会儿也答不上来。容我想想……") # 三个标点，35字以内。

    args = parser.parse_args()
    start = time.time()

    a = GptSovits(args.model_path, args.tokenizer)
    a(args.ref_wav_path, args.text, args.prompt_text, 15)

    print("总耗时：",time.time() - start) 