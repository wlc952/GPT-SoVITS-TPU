# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/models/t2s_model.py
# reference: https://github.com/lifeiteng/vall-e
import torch
from tqdm import tqdm
from typing import List
from AR.modules.embedding_onnx import SinePositionalEmbedding
from AR.modules.embedding_onnx import TokenEmbedding
from AR.modules.transformer_onnx import LayerNorm
from AR.modules.transformer_onnx import TransformerEncoder
from AR.modules.transformer_onnx import TransformerEncoderLayer
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy
from AR.models.utils import topk_sampling
from typing import List


class T2SMLP(nn.Module):
    def __init__(self, w1, b1, w2, b2):
        super(T2SMLP, self).__init__()
        self.linear1 = nn.Linear(w1.size(1), w1.size(0))
        self.linear1.weight = nn.Parameter(w1)
        self.linear1.bias = nn.Parameter(b1)
        self.linear2 = nn.Linear(w2.size(1), w2.size(0))
        self.linear2.weight = nn.Parameter(w2)
        self.linear2.bias = nn.Parameter(b2)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class T2SBlock(nn.Module):
    def __init__(self, num_heads, hidden_dim, mlp, qkv_w, qkv_b, out_w, out_b, norm_w1, norm_b1, norm_eps1, norm_w2, norm_b2, norm_eps2):
        super(T2SBlock, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp = mlp
        
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.qkv.weight = nn.Parameter(qkv_w)
        self.qkv.bias = nn.Parameter(qkv_b)
        
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.out.weight = nn.Parameter(out_w)
        self.out.bias = nn.Parameter(out_b)
        
        self.norm1 = nn.LayerNorm(hidden_dim, eps=norm_eps1)
        self.norm1.weight = nn.Parameter(norm_w1)
        self.norm1.bias = nn.Parameter(norm_b1)
        
        self.norm2 = nn.LayerNorm(hidden_dim, eps=norm_eps2)
        self.norm2.weight = nn.Parameter(norm_w2)
        self.norm2.bias = nn.Parameter(norm_b2)

    def forward(self, x, k_cache, v_cache):
        q, ik, iv = self.qkv(x).chunk(3, dim=-1)

        k_cache = torch.cat([k_cache, ik], dim=1)
        v_cache = torch.cat([v_cache, iv], dim=1)

        kv_len = k_cache.shape[1]
        batch_size = q.shape[0]
        q_len = q.shape[1]

        q = q.view(batch_size, q_len, self.num_heads, -1).transpose(1, 2)
        k = k_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)
        v = v_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.permute(2, 0, 1, 3).reshape(batch_size, -1, self.hidden_dim)

        attn = self.out(attn)

        x = self.norm1(x + attn)
        x = self.norm2(x + self.mlp(x))
        
        return x, ik, iv

    def first(self, x, attn_mask):
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        k_cache = k
        v_cache = v

        kv_len = k_cache.shape[1]
        batch_size = q.shape[0]
        q_len = q.shape[1]

        q = q.view(batch_size, q_len, self.num_heads, -1).transpose(1, 2)
        k = k_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)
        v = v_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask)

        attn = attn.permute(2, 0, 1, 3).reshape(batch_size, -1, self.hidden_dim)

        attn = self.out(attn)

        x = self.norm1(x + attn)
        x = self.norm2(x + self.mlp(x))
        return x, k_cache, v_cache


class T2STransformer(nn.Module):
    def __init__(self, num_blocks: int, blocks: List[nn.Module]):
        super(T2STransformer, self).__init__()
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, k_cache, v_cache, current_size):
        k_list = torch.zeros(24,1,1,512)
        v_list = torch.zeros(24,1,1,512)
        k_cache = k_cache[:, :, : current_size, :]
        v_cache = v_cache[:, :, : current_size, :]

        for i in range(self.num_blocks):
            x, ik, iv = self.blocks[i](x, k_cache[i], v_cache[i])
            k_list[i] = ik
            v_list[i] = iv

        return x, k_list, v_list
    
    def first(self, x, attn_mask):
        k_cache: List[torch.Tensor] = []
        v_cache: List[torch.Tensor] = []
        for block in self.blocks:
            x, k, v = block.first(x, attn_mask)
            k_cache.append(k)
            v_cache.append(v)
        
        current_size = k_cache[0].size(1)
        k_cache_tensor = torch.stack(k_cache, dim=0)
        v_cache_tensor = torch.stack(v_cache, dim=0)
        target_size = 1024
        pad_size = target_size - current_size
        k_cache_tensor = F.pad(k_cache_tensor, (0, 0, 0, pad_size))
        v_cache_tensor = F.pad(v_cache_tensor, (0, 0, 0, pad_size))
        return x, k_cache_tensor, v_cache_tensor, current_size




default_config = {
    "embedding_dim": 512,
    "hidden_dim": 512,
    "num_head": 8,
    "num_layers": 12,
    "num_codebook": 8,
    "p_dropout": 0.0,
    "vocab_size": 1024 + 1,
    "phoneme_vocab_size": 512,
    "EOS": 1024,
}

inf_tensor_value = torch.FloatTensor([-float("Inf")]).float()

def logits_to_probs(
    logits,
    previous_tokens = None,
    temperature: float = 1.0,
    top_k = None,
    top_p = None,
    repetition_penalty: float = 1.0,
):
    previous_tokens = previous_tokens.squeeze()
    if previous_tokens is not None and repetition_penalty != 1.0:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=0, index=previous_tokens)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits.scatter_(dim=0, index=previous_tokens, src=score)

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(
            torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
        )
        sorted_indices_to_remove = cum_probs > top_p
        sorted_indices_to_remove[0] = False  # keep at least one option
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=0, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, -float("Inf"))

    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, top_k)
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, inf_tensor_value, logits)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def multinomial_sample_one_no_sync(
    probs_sort
):  # Does multinomial sampling without a cuda synchronization
    q = torch.randn_like(probs_sort)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample(
    logits,
    previous_tokens,
    **sampling_kwargs,
):
    probs = logits_to_probs(
        logits=logits, previous_tokens=previous_tokens, **sampling_kwargs
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


class OnnxEncoder(nn.Module):
    def __init__(self, ar_text_embedding, bert_proj, ar_text_position):
        super().__init__()
        self.ar_text_embedding = ar_text_embedding
        self.bert_proj = bert_proj
        self.ar_text_position = ar_text_position
    
    def forward(self, x, bert_feature):
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.transpose(1, 2))
        return self.ar_text_position(x)

class FisrtStageDecoder(nn.Module):
    def __init__(self, t2s_transformer_first):
        super().__init__()
        self.t2s_transformer_first = t2s_transformer_first
    def forward(self, xy_pos, xy_attn_mask):
        return self.t2s_transformer_first.first(xy_pos, xy_attn_mask)

class Text2SemanticDecoder(nn.Module):
    def __init__(self, config, norm_first=False, top_k=3):
        super(Text2SemanticDecoder, self).__init__()
        self.model_dim = config["model"]["hidden_dim"]
        self.embedding_dim = config["model"]["embedding_dim"]
        self.num_head = config["model"]["head"]
        self.num_layers = config["model"]["n_layer"]
        self.norm_first = norm_first
        self.vocab_size = config["model"]["vocab_size"]
        self.phoneme_vocab_size = config["model"]["phoneme_vocab_size"]
        self.p_dropout = float(config["model"]["dropout"])
        self.EOS = config["model"]["EOS"]
        self.norm_first = norm_first
        assert self.EOS == self.vocab_size - 1
        self.bert_proj = nn.Linear(1024, self.embedding_dim)
        self.ar_text_embedding = TokenEmbedding(self.embedding_dim, self.phoneme_vocab_size, self.p_dropout)
        self.ar_text_position = SinePositionalEmbedding(self.embedding_dim, dropout=0.1, scale=False, alpha=True)
        self.ar_audio_embedding = TokenEmbedding(self.embedding_dim, self.vocab_size, self.p_dropout)
        self.ar_audio_position = SinePositionalEmbedding(self.embedding_dim, dropout=0.1, scale=False, alpha=True)
        self.h = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=self.model_dim,
                nhead=self.num_head,
                dim_feedforward=self.model_dim * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=norm_first,
            ),
            num_layers=self.num_layers,
            norm=LayerNorm(self.model_dim) if norm_first else None,
        )
        self.ar_predict_layer = nn.Linear(self.model_dim, self.vocab_size, bias=False)
        self.loss_fct = nn.CrossEntropyLoss(reduction="sum")
        self.ar_accuracy_metric = MulticlassAccuracy(
            self.vocab_size,
            top_k=top_k,
            average="micro",
            multidim_average="global",
            ignore_index=self.EOS,
        )
        self.top_k = torch.LongTensor([15])
        self.early_stop_num = torch.LongTensor([-1])



    def init_onnx(self):
        self.onnx_encoder = OnnxEncoder(self.ar_text_embedding, self.bert_proj, self.ar_text_position)
        self.update_next_step = UpdateNextStep(self.ar_audio_embedding, self.ar_audio_position)
        self.sample_layer = SampleLayer()
        self.embedding = Embedding(self.ar_audio_embedding, self.ar_audio_position)
        self.mask = Mask()
        
        blocks = []
        for i in range(self.num_layers):
            layer = self.h.layers[i]
            t2smlp = T2SMLP(
                layer.linear1.weight,
                layer.linear1.bias,
                layer.linear2.weight,
                layer.linear2.bias
            )
            block = T2SBlock(
                self.num_head,
                self.model_dim,
                t2smlp,
                layer.self_attn.in_proj_weight,
                layer.self_attn.in_proj_bias,
                layer.self_attn.out_proj.weight,
                layer.self_attn.out_proj.bias,
                layer.norm1.weight,
                layer.norm1.bias,
                layer.norm1.eps,
                layer.norm2.weight,
                layer.norm2.bias,
                layer.norm2.eps
            )
            blocks.append(block)
        self.t2s_transformer = T2STransformer(self.num_layers, blocks)
        self.first_stage_decoder = FisrtStageDecoder(self.t2s_transformer)


    def infer(self, x, prompts, bert_feature):
        with torch.no_grad():
            top_k = self.top_k
            early_stop_num = self.early_stop_num

            x = self.ar_text_embedding(x)
            x = x + self.bert_proj(bert_feature.transpose(1, 2))
            x = self.ar_text_position(x)

            y = prompts
            x_len = x.shape[1]
            prefix_len = y.shape[1]


            y_emb = self.ar_audio_embedding(y)
            y_len = y_emb.shape[1]
            y_pos = self.ar_audio_position(y_emb)
            xy_pos = torch.concat([x, y_pos], dim=1) 

            # torch.onnx.export(
            #     self.mask,
            #     (torch.tensor([x_len]), torch.tensor([y_len])),
            #     "onnx/cache/mask.onnx",
            #     input_names=["x_len", "y_len"],
            #     output_names=["xy_attn_mask"],
            #     do_constant_folding=True,
            #     opset_version=15,
            # )

            x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)
            x_attn_mask_pad = F.pad(x_attn_mask, (0, y_len), value=True)
            y_attn_mask = F.pad(torch.triu(torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1), (x_len, 0), value=False)
            xy_attn_mask = torch.concat([x_attn_mask_pad, y_attn_mask], dim=0)
            xy_attn_mask = canonical_mask(xy_attn_mask, xy_pos.dtype)
            
            xy_dec, k_cache, v_cache, current_size = self.t2s_transformer.first(xy_pos, xy_attn_mask)

            print("x_len:",x_len,"y_len:",y_len)
            print("current_size",current_size)

            logits = self.ar_predict_layer(xy_dec[:, -1])
            logits = logits[:, :-1]

            # torch.onnx.export(
            #     self.sample_layer,
            #     (logits[0], y),
            #     "onnx/cache/sample_layer.onnx",
            #     input_names=["logits", "y"],
            #     output_names=["samples", "yy"],
            #     dynamic_axes={"logits": {0: "log"}, "y": {1: "seq"}},
            #     do_constant_folding=True,
            #     opset_version=15,
            # )
            samples = sample(logits[0], y, top_k=top_k, top_p=1.0, repetition_penalty=1.35)[0].unsqueeze(0)
            y = torch.concat([y, samples], dim=1)

            y_emb = self.ar_audio_embedding(y[:, -1:])
            xy_pos = y_emb * self.ar_audio_position.x_scale + self.ar_audio_position.alpha * self.ar_audio_position.pe[:, y_len].to(dtype=y_emb.dtype,device=y_emb.device)


            # torch.onnx.export(
            #     self.t2s_transformer,
            #     (xy_pos, k_cache, v_cache, torch.tensor([current_size])),
            #     "onnx/cache/t2s_transformer.onnx",
            #     input_names=["xy_pos", "ik", "iv", "current_size"],
            #     output_names=["xy_dec", "k", "v"],
            #     do_constant_folding=True,
            #     opset_version=15,
            # )

            for idx in tqdm(range(1,500)):
                xy_dec, k, v = self.t2s_transformer(xy_pos, k_cache, v_cache, current_size)
                k_cache[:, :, current_size: current_size+1, :] = k
                v_cache[:, :, current_size: current_size+1, :] = v

                current_size = current_size + 1
                logits = self.ar_predict_layer(xy_dec[:, -1])
                samples = sample(logits[0], y, top_k=top_k, top_p=1.0, repetition_penalty=1.35)[0].unsqueeze(0)
                y = torch.concat([y, samples], dim=1)

                if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num: break
                if torch.argmax(logits, dim=-1)[0] == self.EOS or samples[0, 0] == self.EOS: break

                y_emb = self.ar_audio_embedding(y[:, -1:])
                xy_pos = y_emb * self.ar_audio_position.x_scale + self.ar_audio_position.alpha * self.ar_audio_position.pe[:, y_len + idx].to(dtype=y_emb.dtype,device=y_emb.device)

            print("y", y)

        return y[:, :-1], idx - 1
      

    def forward(self, x, prompts, bert_feature):

        x = self.onnx_encoder(x, bert_feature)
        y = prompts
        x_len = x.shape[1]
        y_len = y.shape[1]
        xy_pos = self.embedding(x, y)
        xy_attn_mask = self.mask(x_len, y_len)

        xy_dec, k_cache, v_cache, current_size = self.first_stage_decoder(xy_pos, xy_attn_mask)
        logits = self.ar_predict_layer(xy_dec[:, -1])
        logits = logits[:, :-1]
        samples, y = self.sample_layer(logits[0], y)
        xy_pos = self.update_next_step(y[:, -1:], y_len)

        for idx in tqdm(range(1,300)):
            xy_dec, k_cache, v_cache, current_size = self.t2s_transformer(xy_pos, k_cache, v_cache, current_size)
            logits = self.ar_predict_layer(xy_dec[:, -1])
            samples, y = self.sample_layer(logits[0], y)

            if torch.argmax(logits, dim=-1)[0] == self.EOS or samples[0, 0] == self.EOS:
                break

            xy_pos = self.update_next_step(y[:, -1:], idx + y_len)

        return y[:, :-1], idx - 1




class Mask(nn.Module):
    def __init__(self, ):
        super().__init__()
    def forward(self, x_len,y_len):
        x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)
        x_attn_mask_pad = F.pad(x_attn_mask, (0, y_len), value=True)
        y_attn_mask = F.pad(torch.triu(torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1), (x_len, 0), value=False)
        xy_attn_mask = torch.concat([x_attn_mask_pad, y_attn_mask], dim=0)
        xy_attn_mask = canonical_mask(xy_attn_mask, torch.float32)
        return xy_attn_mask


class Embedding(nn.Module):
    def __init__(self, ar_audio_embedding, ar_audio_position):
        super().__init__()
        self.ar_audio_embedding = ar_audio_embedding
        self.ar_audio_position = ar_audio_position
    def forward(self, x, y):
        y_emb = self.ar_audio_embedding(y)
        y_pos = self.ar_audio_position(y_emb)
        xy_pos = torch.concat([x, y_pos], dim=1) 
        return xy_pos
    
 
class SampleLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, logits, y):
        samples = sample(logits, y, top_k=15, top_p=1.0, repetition_penalty=1.35)[0].unsqueeze(0)
        y = torch.concat([y, samples], dim=1)
        return samples, y
    

class UpdateNextStep(nn.Module):
    def __init__(self, ar_audio_embedding, ar_audio_position):
        super().__init__()
        self.ar_audio_embedding = ar_audio_embedding
        self.ar_audio_position = ar_audio_position
    def forward(self, y, idx_plus_len):
        y_emb = self.ar_audio_embedding(y)
        xy_pos = y_emb * self.ar_audio_position.x_scale + self.ar_audio_position.alpha * self.ar_audio_position.pe[:, idx_plus_len]
        return xy_pos


def canonical_mask(mask,target_type):
    if mask is not None:
        _mask_dtype = mask.dtype
        _mask_is_float = torch.is_floating_point(mask)
        if _mask_dtype != torch.bool and not _mask_is_float:
            raise AssertionError(
                "only bool and floating types are supported")
        if not _mask_is_float:
            mask = (
                torch.zeros_like(mask, dtype=target_type)
                .masked_fill_(mask, float("-100000"))
            )
    return mask


