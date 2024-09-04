import os
import sys
now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append(now_dir + "/GPT_SoVITS")

from tqdm import tqdm
from transformers import HubertModel, AutoModelForMaskedLM, AutoTokenizer
from module.models_onnx import SynthesizerTrn
from AR.models.t2s_lightning_module_onnx import Text2SemanticLightningModule
from torch import nn
import torch
import numpy as np
import librosa
import re
from inference_webui import spectrogram_torch, DictToAttrRecursive, get_first, load_audio
from text.chinese2 import text_normalize
from text import cleaned_text_to_sequence
from text.symbols2 import punctuation
from text.chinese2 import g2p
import time
import soundfile
import torch.nn.functional as F

splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }


def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    # audio,sr = librosa.load(filename, sr=int(hps.data.sampling_rate))

    audio = torch.FloatTensor(audio)

    maxx=audio.abs().max()
    if(maxx>1):audio/=min(2,maxx)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec

class SSLModel(nn.Module):
    def __init__(self,cnhubert_base_path):
        super().__init__()
        self.model = HubertModel.from_pretrained(cnhubert_base_path, local_files_only=True)
    
    def prepare_input(self, ref_wav_path):
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
            print("音频长度不符合要求")
            raise ValueError
        
        wav16k = torch.from_numpy(wav16k)
        zero_wav = np.zeros(int(32000 * 0.3),dtype=np.float32)
        zero_wav_torch = torch.from_numpy(zero_wav)
        wav16k = torch.cat([zero_wav_torch, wav16k, zero_wav_torch])
        return wav16k.unsqueeze(0), zero_wav

    def forward(self, ref_audio_16k):
        self.model.eval()
        ssl_content = self.model(ref_audio_16k)["last_hidden_state"].transpose(1, 2)
        return ssl_content

    def export(self, ref_audio_16k, project_name):
        torch.onnx.export(
            self,
            (ref_audio_16k),
            f"onnx/{project_name}/00_cnhubert.onnx",
            input_names=["ref_audio_16k"],
            output_names=["last_hidden_state"],
            dynamic_axes={"ref_audio_16k": {1: "seq"}},
            do_constant_folding=True,
            opset_version=15,
            verbose=False
        )
        

class VitsModel(nn.Module):
    def __init__(self, vits_path):
        super().__init__()
        dict_s2 = torch.load(vits_path,map_location="cpu")
        self.hps = dict_s2["config"]
        self.hps = DictToAttrRecursive(self.hps)
        self.hps.model.semantic_frame_rate = "25hz"
        self.vq_model = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model
        )
        self.vq_model.eval()
        self.vq_model.load_state_dict(dict_s2["weight"], strict=False)

    def forward(self, pred_semantic, text_seq, refer):
        return self.vq_model(pred_semantic, text_seq, refer)
    
    def export(self, pred_semantic, text_seq, refer, project_name):
        torch.onnx.export(
            self,
            (pred_semantic, text_seq, refer),
            f"onnx/{project_name}/09_vits_decoder.onnx",
            input_names=["pred_semantic", "text_seq", "refer"],
            output_names=["audio"],
            dynamic_axes={"pred_semantic": {2: "semantic_seq"}, "text_seq": {1: "text_seq"}, "refer": {2: "refer_seq"}},
            do_constant_folding=True,
            opset_version=15,
            verbose=False
        )


class VitsEncoder(nn.Module):
    def __init__(self, vits):
        super().__init__()
        self.vits = vits.vq_model
    
    def forward(self, ssl_content):
        codes = self.vits.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
        prompt = prompt_semantic.unsqueeze(0)
        return prompt
    
    def export(self, ssl_content, project_name):
        torch.onnx.export(
            self,
            ssl_content,
            f"onnx/{project_name}/01_vits_encoder.onnx",
            input_names=["ssl_content"],
            output_names=["prompt_semantic"],
            dynamic_axes={"ssl_content": {2: "seq"}},
            do_constant_folding=True,
            opset_version=15,
            verbose=False
        )
    

class BertModel(nn.Module):
    def __init__(self,bert_path):
        super().__init__()
        self.bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
        self.bert_model.eval()

    def forward(self, input_ids, token_type_ids, attention_mask):
        with torch.no_grad():
            res = self.bert_model(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0][1:-1]       
        return res
    
    def export(self, input_ids, token_type_ids, attention_mask, project_name):
        torch.onnx.export(
            self,
            (input_ids, token_type_ids, attention_mask),
            f"onnx/{project_name}/02_bert.onnx",
            input_names=["input_ids", "token_type_ids", "attention_mask"],
            output_names=["hidden_states"],
            dynamic_axes={"input_ids": {1: "seq"}, "token_type_ids": {1: "seq"}, "attention_mask": {1: "seq"}},
            do_constant_folding=True,
            opset_version=15,
        )


class G2PWBertModel(nn.Module):
    def __init__(self,bert_path):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
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

    def get_bert(self, text, word2ph):
        inputs = self.tokenizer(text, return_tensors="pt")
        a, b, c = inputs["input_ids"], inputs["token_type_ids"], inputs["attention_mask"]

        res = self.bert_model(a, b, c)

        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T
    
    def forward(self, text, prompt_text):
        norm_text, norm_prompt_text = self.prepare_input(text, prompt_text)

        print("处理后的参考文本:", norm_prompt_text, len(norm_prompt_text))
        print("处理后的目标文本:", norm_text, len(norm_text))

        phones1, word2ph1 = g2p(norm_text) #g2pw模型
        phones1 = cleaned_text_to_sequence(phones1, "v2")
        bert1 = self.get_bert(norm_text, word2ph1) #BERT模型

        phones2, word2ph2 = g2p(norm_prompt_text) #g2pw模型
        phones2 = cleaned_text_to_sequence(phones2, "v2")

        bert2 = self.get_bert(norm_prompt_text, word2ph2) #BERT模型

        bert = torch.cat([bert2, bert1], 1).unsqueeze(0)

        all_phoneme_ids = phones2 + phones1
        all_phoneme_ids = torch.LongTensor(all_phoneme_ids).unsqueeze(0)
        return bert, all_phoneme_ids, phones1
    
    def export(self, text, prompt_text, project_name):
        norm_text, norm_prompt_text = self.prepare_input(text, prompt_text)
        phones1, word2ph1 = g2p(norm_text) #g2pw模型
        phones1 = cleaned_text_to_sequence(phones1, "v2")
        inputs = self.tokenizer(norm_text, return_tensors="pt")
        a, b, c = inputs["input_ids"], inputs["token_type_ids"], inputs["attention_mask"]
        self.bert_model.export(a, b, c, project_name)


class T2SModel(nn.Module):
    def __init__(self, t2s_path):
        super().__init__()
        dict_s1 = torch.load(t2s_path, map_location="cpu")
        self.config = dict_s1["config"]
        self.t2s_model = Text2SemanticLightningModule(self.config, "****", is_train=False)
        self.t2s_model.load_state_dict(dict_s1["weight"])
        self.t2s_model.eval()
        self.hz = 50
        self.max_sec = self.config["data"]["max_sec"]
        self.t2s_model.model.top_k = torch.LongTensor([15])
        self.t2s_model.model.early_stop_num = torch.LongTensor([self.hz * self.max_sec])
        self.t2s_model = self.t2s_model.model #Text2SemanticDecoder
        self.EOS = self.t2s_model.EOS
        self.t2s_model.init_onnx()
        self.t2s_encoder = self.t2s_model.onnx_encoder
        self.first_stage_decoder = self.t2s_model.first_stage_decoder
        self.stage_decoder = self.t2s_model.stage_decoder

    def forward(self, phoneme_ids, bert, prompts):
        x = self.t2s_encoder(phoneme_ids, bert)
        y, k, v, y_emb, x_example = self.first_stage_decoder(x, prompts)

        stop = False
        for idx in tqdm(range(1, 300)):
            enco = self.stage_decoder(y, k, v, y_emb, x_example)
            y, k, v, y_emb, logits, samples = enco

            if torch.argmax(logits, dim=-1)[0] == self.EOS or samples[0, 0] == self.EOS:
                stop = True
            if stop: break

        return y[...,-idx:-1].unsqueeze(0)

    def export(self, phoneme_ids, bert, prompts, project_name):
        torch.onnx.export(
            self.t2s_encoder,
            (phoneme_ids, bert),
            f"onnx/{project_name}/03_t2s_encoder.onnx",
            input_names=["phoneme_ids", "bert"],
            output_names=["x"],
            dynamic_axes={"phoneme_ids": {1: "seq"}, "bert": {2: "seq"}},
            do_constant_folding=True,
            opset_version=15
        )
        x = self.t2s_encoder(phoneme_ids, bert)
        torch.onnx.export(
            self.first_stage_decoder,
            (x, prompts),
            f"onnx/{project_name}/04_t2s_first_stage_decoder.onnx",
            input_names=["x", "prompts"],
            output_names=["y", "k", "v", "y_emb", "x_example"],
            dynamic_axes={"x": {1 : "x_length"}, "prompts": {1 : "prompts_length"}},
            do_constant_folding=True,
            opset_version=15
        )
        y, k, v, y_emb, x_example = self.first_stage_decoder(x, prompts)
        torch.onnx.export(
            self.stage_decoder,
            (y, k, v, y_emb, x_example),
            f"onnx/{project_name}/05_t2s_stage_decoder.onnx",
            input_names=["iy", "ik", "iv", "iy_emb", "x_example"],
            output_names=["y", "k", "v", "y_emb", "logits", "samples"],
            dynamic_axes={
                "iy": {1 : "iy_length"},
                "ik": {1 : "ik_length"},
                "iv": {1 : "iv_length"},
                "iy_emb": {1 : "iy_emb_length"},
                "ix_example": {1 : "ix_example_length"},
            },
            do_constant_folding=True,
            opset_version=15
        )

     
    
    


class GptSovits(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ssl_model = SSLModel(args.ssl_path)
        self.bert_model = G2PWBertModel(args.bert_path)
        self.vits_model = VitsModel(args.vits_path)
        self.vits_encoder = VitsEncoder(self.vits_model)
        self.t2s_model = T2SModel(args.gpt_path)
    
    def forward(self, args):
        ref_wav_16k, zero_wav = self.ssl_model.prepare_input(args.ref_wav_path)

        # self.ssl_model.export(ref_wav_16k, args.project_name)
        ssl_content = self.ssl_model(ref_wav_16k)

        # self.vits_encoder.export(ssl_content, args.project_name)
        prompt = self.vits_encoder(ssl_content)

        # self.bert_model.export(args.text, args.prompt_text, args.project_name)
        bert, all_phoneme_ids, phones1 = self.bert_model(args.text, args.prompt_text)

        # self.t2s_model.export(all_phoneme_ids, bert, prompt, args.project_name)
        pred_semantic = self.t2s_model(all_phoneme_ids, bert, prompt)
        refers = get_spepc(self.vits_model.hps, args.ref_wav_path)

        # self.vits_model.export(pred_semantic, torch.LongTensor(phones1).unsqueeze(0), refers, args.project_name)
        audio = self.vits_model(pred_semantic, torch.LongTensor(phones1).unsqueeze(0), refers).detach().cpu().numpy()[0, 0]

        max_audio=np.abs(audio).max()
        if max_audio>1:audio/=max_audio

        audio_opt = []
        audio_opt.append(audio)
        audio_opt.append(zero_wav)

        audio = (np.concatenate(audio_opt, 0) * 32768).astype(np.int16)

        soundfile.write("out.wav", audio, self.vits_model.hps.data.sampling_rate)

        return audio
        



if __name__ == "__main__":
    try:
        os.mkdir("onnx")
    except:
        pass
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssl_path", type=str, default="GPT_SoVITS/pretrained_models/chinese-hubert-base")
    parser.add_argument("--bert_path", type=str, default="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large")
    parser.add_argument("--vits_path", type=str, default="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth")
    parser.add_argument("--gpt_path", type=str, default="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt")
   
    parser.add_argument("--ref_wav_path", type=str, default="参考音频/说话-正是像停云小姐这样的接渡使往来周旋，仙舟的贸易才能有如今的繁盛。.wav")
    parser.add_argument("--text", type=str, default="大家好，我是来自四川理塘的王真先生。") #三个标点符号。首句少于四个字，两个标点符号。
    parser.add_argument("--prompt_text", type=str, default="。正是像停云小姐这样的接渡使往来周旋，仙舟的贸易才能有如今的繁盛。") #三个标点符号

    parser.add_argument("--project_name", type=str, default="dynamic")
    args = parser.parse_args()

    try:
        os.mkdir(f"onnx/{args.project_name}")
    except:
        pass

    gptsovits = GptSovits(args)
    a = gptsovits(args)






