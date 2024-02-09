######################################################################################
# The implementation relies on http://nlp.seas.harvard.edu/2018/04/03/attention.html #
######################################################################################

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Identity(nn.Module):
    
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
class VocabularyEmbedder(nn.Module):
    
    def __init__(self, voc_size, d_model):
        super(VocabularyEmbedder, self).__init__()
        self.voc_size = voc_size
        self.d_model = d_model
        self.embedder = nn.Embedding(voc_size, d_model) #寻找语义关系！！！！！！！！！！！！！！！！！！！！
        
    def forward(self, x): # x - tokens (B, seq_len)
        x = self.embedder(x)
        # print(x.shape)
        x = x * np.sqrt(self.d_model)
        
        return x # (B, seq_len, d_model)
    
class FeatureEmbedder(nn.Module):
    
    def __init__(self, d_feat, d_model):
        super(FeatureEmbedder, self).__init__()
        self.d_model = d_model
        self.embedder = nn.Linear(d_feat, d_model)
        
    def forward(self, x): # x - tokens (B, seq_len, d_feat)
        x = self.embedder(x)
        x = x * np.sqrt(self.d_model)
        
        return x # (B, seq_len, d_model)
    
class PositionalEncoder(nn.Module):
    
    def __init__(self, d_model, dout_p, seq_len=3660): # 3651 max feat len for c3d
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dout_p)   #剪枝的trick
        
        pos_enc_mat = np.zeros((seq_len, d_model))
        odds = np.arange(0, d_model, 2)
        evens = np.arange(1, d_model, 2)

        for pos in range(seq_len):
            pos_enc_mat[pos, odds] = np.sin(pos / (10000 ** (odds / d_model)))
            pos_enc_mat[pos, evens] = np.cos(pos / (10000 ** (evens / d_model)))
        
        self.pos_enc_mat = torch.from_numpy(pos_enc_mat).unsqueeze(0)
        
    def forward(self, x): # x - embeddings (B, seq_len, d_model)
        B, S, d_model = x.shape
        # torch.cuda.FloatTensor torch.FloatTensor
        x = x + self.pos_enc_mat[:, :S, :].type_as(x)
        x = self.dropout(x)
        
        return x # same as input

def subsequent_mask(size):
    mask = torch.ones(1, size, size)
    mask = torch.tril(mask, 0)
    
    return mask.byte() # ([1, size, size])

def mask(src, trg, pad_idx):
    # masking the padding. src shape: (B, S') -> (B, 1, S')
    src_mask = (src != pad_idx).unsqueeze(1)
    
    if trg is not None:
        trg_mask = (trg != pad_idx).unsqueeze(-2) & subsequent_mask(trg.size(-1)).type_as(src_mask.data)

        return src_mask, trg_mask
    
    else:
        return src_mask


def mask_out(src, trg, pad_idx):
    # masking the padding. src shape: (B, S') -> (B, 1, S')
    src_mask = (src != pad_idx).unsqueeze(1)

    if trg is not None:
        trg_mask = (trg != pad_idx).unsqueeze(-2) & subsequent_mask(trg.size(-1)).type_as(src_mask.data)

        return src_mask, trg_mask

    else:
        return src_mask
def clone(module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])

def attention(Q, K, V, mask):
    # Q, K, V are # (B, *(H), seq_len, d_model//H = d_k)
    d_k = Q.size(-1)        #256/1024/512
    # print("att",Q.shape,K.shape)    #torch.Size([32, 4, 33, 256]) torch.Size([32, 4, 155, 256])
    QKt = Q.matmul(K.transpose(-1, -2)) #矩阵相乘[24, 256]和[203, 256]的转置矩阵点乘
    sm_input = QKt / np.sqrt(d_k)
    # print(d_k)
    # print("QKt", QKt.shape)
    # print("mask",mask.shape)
    if mask is not None:
        sm_input = sm_input.masked_fill(mask == 0, -float('inf'))
    # print("sm_input", sm_input.shape)
    softmax = F.softmax(sm_input, dim=-1)   #尺寸不调整
    # print("softmax", softmax.shape)
    out = softmax.matmul(V)         #恢复之前的尺寸，但是已经把数据融入进去了
    # print("out1", out.shape)
    return out # (B, *(H), seq_len, d_model//H = d_k)

def Fusionattention(Q, K, V, mask):
    # Q, K, V are # (B, *(H), seq_len, d_model//H = d_k)
    d_k = K.size(-1)        #256/1024/512
    # print(K.shape, Q.shape)

    QKt = Q.transpose(-1, -2).matmul(K) #矩阵相乘[202, 1024]和[202, 1024+128]的转置矩阵点乘
    sm_input = QKt / np.sqrt(d_k)

    # print(d_k)
    # print("sm_input", sm_input.shape, "d_k",d_k, "QKt", QKt.shape)

    # mask = torch.ones((32, 1, 1, 288), device=0)
    # print("mask0", mask.shape)
    # mask = mask[:, : , : , :]
    # # print("mask1", mask.shape)
    # mask = (1.0 - mask) * -1e9
    #
    # if mask is not None:
    #    sm_input = sm_input.masked_fill(mask == 0, -float('inf'))
    # print("mask2", mask.shape)
    # print("sm_input", sm_input.shape)
    softmax = F.softmax(sm_input, dim=-1)   #尺寸不调整
    # print("softmax", softmax.shape)
    out = (softmax.matmul(V.transpose(-1, -2))).transpose(-1, -2)       #恢复之前的尺寸，但是已经把数据融入进去了
    # print("out", out.shape)

    return out # (B, *(H), seq_len, d_model//H = d_k)


class FusionMultiheadedAttention(nn.Module):  # 多头注意力  在其中加入依存文本

    def __init__(self, d_model, H):
        super(FusionMultiheadedAttention, self).__init__()
        assert d_model % H == 0
        self.d_model = d_model
        self.H = H
        self.d_k = d_model // H
        self.d_kv = (d_model+128) // H
        # Q, K, V, and after-attention layer (4). out_features is d_model
        # because we apply linear at all heads at the same time
        self.linears = clone(nn.Linear(d_model, d_model), 4)  # bias True??
        self.linears_x = clone(nn.Linear(d_model, 960), 4)  # bias True??
        self.linears_y = clone(nn.Linear(128, 64), 4)  # bias True??
        self.linears_KV = clone(nn.Linear(1024+128, 1152), 4)  # bias True??

    def forward(self, XQ, XK, XV, YK ,YV, mask):  # Q, K, V are of size (B, seq_len, d_model)

        B, seq_len, d_model = XQ.shape
        Q = self.linears[0](XQ)  # (B, *, in_features) -> (B, *, out_features)
        #XK = self.linears_x[0](XK)  # (B, *, in_features) -> (B, *, out_features)
        #XV = self.linears_x[1](XV)  # (B, *, in_features) -> (B, *, out_features)
        #YK = self.linears_y[0](YK)
       # YV = self.linears_y[1](YV)
        K = torch.cat((XK, YK), 2)
        V = torch.cat((XV, YV), 2)
        # print("XV", XV.shape, "YV", YV.shape)
        K = self.linears_KV[0](K)
        V = self.linears_KV[1](V)
        K = F.leaky_relu(K)
        V = F.leaky_relu(V)
        # print("V", V.shape, "K", K.shape)
        K = F.log_softmax(K, dim = -1)
        V = F.log_softmax(V, dim = -1)
        # print("V", V.shape, "K", K.shape)
        # print("Q:", Q.shape)
        # print("K:", K.shape)
        # print("V:", V.shape)
        # print("d_k",self.d_k, "d_kv",self.d_kv,"h",self.H)
        Q = Q.view(B, -1, self.H, self.d_k).transpose(-3, -2)  # (-4, -3*, -2*, -1)
        K = K.view(B, -1, self.H, self.d_kv).transpose(-3, -2)  # view( )相当于reshape、resize，对Tensor的形状进行调整
        V = V.view(B, -1, self.H, self.d_kv).transpose(-3, -2)   #分为H个头，H=4
        # print("Q:", Q.shape)
        # print("K:", K.shape)
        # print("V:", V.shape)
        if mask is not None:
            # the same mask for all heads
            mask = mask.unsqueeze(1)

        # todo: check whether they are both calculated here and how can be
        # serve both.
        att = Fusionattention(Q, K, V, mask)  # (B, H, seq_len, d_k)
        # print("att1", att.shape)
        att = att.transpose(-3, -2).contiguous().view(B, seq_len, d_model)
        # print("att", att.shape)
        att = self.linears[3](att)

        return att  # (B, H, seq_len, d_k)
class MultiheadedAttention(nn.Module):          #多头注意力  在其中加入依存文本
    
    def __init__(self, d_model, H):
        super(MultiheadedAttention, self).__init__()
        assert d_model % H == 0
        self.d_model = d_model
        self.H = H
        self.d_k = d_model // H
        # Q, K, V, and after-attention layer (4). out_features is d_model
        # because we apply linear at all heads at the same time
        self.linears = clone(nn.Linear(d_model, d_model), 4) # bias True??
        
    def forward(self, Q, K, V, mask): # Q, K, V are of size (B, seq_len, d_model)
        B, seq_len, d_model = Q.shape
        
        Q = self.linears[0](Q) # (B, *, in_features) -> (B, *, out_features)
        K = self.linears[1](K)
        V = self.linears[2](V)

        # print("V-1", V.shape, "K-1", K.shape)
        Q = Q.view(B, -1, self.H, self.d_k).transpose(-3, -2) # (-4, -3*, -2*, -1)
        K = K.view(B, -1, self.H, self.d_k).transpose(-3, -2)#view( )相当于reshape、resize，对Tensor的形状进行调整
        V = V.view(B, -1, self.H, self.d_k).transpose(-3, -2)
        # print("Q:", Q.shape)
        # print("K:", K.shape)
        # print("V:", V.shape)
        if mask is not None:
            # the same mask for all heads
            mask = mask.unsqueeze(1)
        
        # todo: check whether they are both calculated here and how can be 
        # serve both.
        att = attention(Q, K, V, mask) # (B, H, seq_len, d_k)
        att = att.transpose(-3, -2).contiguous().view(B, seq_len, d_model)
        # print("att1", att.shape)
        att = self.linears[3](att)

        return att # (B, H, seq_len, d_k)
    
class ResidualConnection(nn.Module):
    
    def __init__(self, size, dout_p):
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dout_p)
        
    def forward(self, x, sublayer): # [(B, seq_len, d_model), attention or feed forward]
        res = self.norm(x)
        res = sublayer(res)
        res = self.dropout(res)
        
        return x + res
    
class PositionwiseFeedForward(nn.Module):
    
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        # todo dropout?
        
    def forward(self, x): # x - (B, seq_len, d_model)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x # x - (B, seq_len, d_model)
    
class EncoderLayer(nn.Module):
    
    def __init__(self, d_model, dout_p, H, d_ff):
        super(EncoderLayer, self).__init__()
        self.res_layers = clone(ResidualConnection(d_model, dout_p), 2)
        self.self_att = MultiheadedAttention(d_model, H)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        
    def forward(self, x, src_mask): # x - (B, seq_len, d_model) src_mask (B, 1, S)
        # sublayer should be a function which inputs x and outputs transformation
        # thus, lambda is used instead of just `self.self_att(x, x, x)` which outputs 
        # the output of the self attention
        sublayer0 = lambda x: self.self_att(x, x, x, src_mask)
        sublayer1 = self.feed_forward
        
        x = self.res_layers[0](x, sublayer0)
        x = self.res_layers[1](x, sublayer1)
        
        return x # x - (B, seq_len, d_model)
    
class Encoder(nn.Module):
    
    def __init__(self, d_model, dout_p, H, d_ff, N):
        super(Encoder, self).__init__()
        self.enc_layers = clone(EncoderLayer(d_model, dout_p, H, d_ff), N)
        # print("n",N)          #N=1为几层Encoder
        
    def forward(self, x, src_mask): # x - (B, seq_len, d_model) src_mask (B, 1, S)
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        
        return x # x - (B, seq_len, d_model) which will be used as Q and K in decoder
    
class DecoderLayer(nn.Module):
    
    def __init__(self, d_model, dout_p, H, d_ff):
        super(DecoderLayer, self).__init__()
        self.res_layers = clone(ResidualConnection(d_model, dout_p), 3)
        self.self_att = MultiheadedAttention(d_model, H)
        self.enc_att = MultiheadedAttention(d_model, H)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        # TODO: 
        # should all multiheaded and feed forward
        # attention be the same, check the parameter number
        
    def forward(self, x, memory, src_mask, trg_mask): # x, memory - (B, seq_len, d_model) src_mask (B, 1, S) trg_mask (B, S, S)
        # a comment regarding the motivation of the lambda function 
        # please see the EncoderLayer
        sublayer0 = lambda x: self.self_att(x, x, x, trg_mask)
        # print("\n\nenc_att:\n")
        sublayer1 = lambda x: self.enc_att(x, memory, memory, src_mask)
        sublayer2 = self.feed_forward
        
        x = self.res_layers[0](x, sublayer0)
        x = self.res_layers[1](x, sublayer1)
        x = self.res_layers[2](x, sublayer2)
        
        return x # x, memory - (B, seq_len, d_model)
    
class Decoder(nn.Module):
    
    def __init__(self, d_model, dout_p, H, d_ff, N):
        super(Decoder, self).__init__()
        self.dec_layers = clone(DecoderLayer(d_model, dout_p, H, d_ff), N)
        
    def forward(self, x, memory, src_mask, trg_mask): # x (B, S, d_model) src_mask (B, 1, S) trg_mask (B, S, S)
        for layer in self.dec_layers:
            x = layer(x, memory, src_mask, trg_mask)
        # todo: norm?
        return x # (B, S, d_model)


class FusionLayer(nn.Module):
    def __init__(self, d_model, dout_p, H, d_ff):
        super(FusionLayer, self).__init__()
        self.res_layers = clone(ResidualConnection(d_model, dout_p), 2)
        self.self_att = FusionMultiheadedAttention(d_model, H)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)

    def forward(self, x, y,src_mask):  # x - (B, seq_len, d_model) src_mask (B, 1, S)
        # sublayer should be a function which inputs x and outputs transformation
        # thus, lambda is used instead of just `self.self_att(x, x, x)` which outputs
        # the output of the self attention
        # print(x.shape, y.shape)
        # XYsum = torch.cat((x, y), 2)
        # print("XYsum", XYsum.shape)
        sublayer0 = lambda x: self.self_att(x, x, x, y, y, src_mask)
        sublayer1 = self.feed_forward

        x = self.res_layers[0](x, sublayer0)
        x = self.res_layers[1](x, sublayer1)

        return x  # x - (B, seq_len, d_model)


class Fusioner(nn.Module):

    def __init__(self, d_model, dout_p, H, d_ff, N):
        super(Fusioner, self).__init__()
        self.enc_layers = clone(FusionLayer(d_model, dout_p, H, d_ff), N)
        # print("n",N)          #N=1为几层Encoder

    def forward(self, x, y, src_mask):  # x - (B, seq_len, d_model) src_mask (B, 1, S)
        for layer in self.enc_layers:
            x = layer(x, y, src_mask)

        return x  # x - (B, seq_len, d_model) which will be used as Q and K in decoder



class SubsAudioVideoGeneratorConcatLinearDoutLinear(nn.Module):     #多模态混合
    
    def __init__(self, d_model_subs, d_model_audio, d_model_video, voc_size, dout_p):
        super(SubsAudioVideoGeneratorConcatLinearDoutLinear, self).__init__()
        self.linear = nn.Linear(d_model_subs + d_model_audio + d_model_video, voc_size)
        self.dropout = nn.Dropout(dout_p)
        self.linear2 = nn.Linear(voc_size, voc_size)
        print('using SubsAudioVideoGeneratorConcatLinearDoutLinear')
        
    # ?(B, seq_len, d_model_audio), ?(B, seq_len, d_model_audio), ?(B, seq_len, d_model_video)
    def forward(self, subs_x, audio_x, video_x):
        x = torch.cat([subs_x, audio_x, video_x], dim=-1)
        x = self.linear(x)
        x = self.linear2(self.dropout(F.relu(x)))
        
        return F.log_softmax(x, dim=-1) # (B, seq_len, voc_size)


class ImgsSubsAudioVideoGeneratorConcatLinearDoutLinear(nn.Module):  # 多模态混合

    def __init__(self, d_model_subs, d_model_imgs, d_model_audio, d_model_video, voc_size, dout_p):
        super(ImgsSubsAudioVideoGeneratorConcatLinearDoutLinear, self).__init__()
        self.linear = nn.Linear(d_model_subs + d_model_imgs + d_model_audio + d_model_video, voc_size)
        self.dropout = nn.Dropout(dout_p)
        self.linear2 = nn.Linear(voc_size, voc_size)
        print('using SubsAudioVideoGeneratorConcatLinearDoutLinear')

    # ?(B, seq_len, d_model_audio), ?(B, seq_len, d_model_audio), ?(B, seq_len, d_model_video)
    def forward(self, subs_x, imgs_x,audio_x, video_x):
        x = torch.cat([subs_x, imgs_x,audio_x, video_x], dim=-1)
        # print("x1",x.shape)         #x2 x1 torch.Size([64, 76, 2176])
        x = self.linear(x)
        x = self.linear2(self.dropout(F.relu(x)))
        # print("x2", x.shape)        #x2 torch.Size([64, 76, 10173])
        return F.log_softmax(x, dim=-1)  # (B, seq_len, voc_size)
def normalize_embeddings(a, eps=1e-8):
    a_n = a.norm(dim=1)[:, None]            #中间层取模
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))     #torch.ones_like生成与a_n形状相同的0向量
    return a_norm


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a = normalize_embeddings(a, eps)
    b = normalize_embeddings(b, eps)

    sim_mt = torch.mm(a, b.transpose(0, 1))     #torch.mm是两个矩阵相乘，即两个二维的张量相乘
    return sim_mt



class SubsAudioVideoTransformer(nn.Module): #模型
    
    def __init__(self, trg_voc_size, src_subs_voc_size,src_imgs_voc_size,
                 d_feat_audio, d_feat_video,
                 d_model_audio, d_model_video, d_model_subs,d_model_imgs,
                 d_ff_audio, d_ff_video, d_ff_subs, d_ff_imgs,
                 N_audio, N_video, N_subs,
                 dout_p, H, use_linear_embedder):
        super(SubsAudioVideoTransformer, self).__init__()
        self.src_emb_subs = VocabularyEmbedder(src_subs_voc_size, d_model_subs)
        self.src_emb_imgs = VocabularyEmbedder(src_imgs_voc_size, d_model_imgs)
        if use_linear_embedder:
            self.src_emb_audio = FeatureEmbedder(d_feat_audio, d_model_audio)
            self.src_emb_video = FeatureEmbedder(d_feat_video, d_model_video)
            self.src_emb_video_memory = FeatureEmbedder(d_feat_video, d_model_video)
        else:
            assert d_feat_video == d_model_video and d_feat_audio == d_model_audio
            self.src_emb_audio = Identity()
            self.src_emb_video = Identity()
            self.src_emb_video_memory = Identity()
        self.trg_voc_size = trg_voc_size
        self.trg_emb_subs  = VocabularyEmbedder(trg_voc_size, d_model_subs) #目标文本
        self.trg_emb_imgs  = VocabularyEmbedder(trg_voc_size, d_model_imgs)  # 目标文本
        self.trg_emb_audio = VocabularyEmbedder(trg_voc_size, d_model_audio)
        self.trg_emb_video = VocabularyEmbedder(trg_voc_size, d_model_video)
        self.trg_emb_memory = VocabularyEmbedder(trg_voc_size, 128)
        # self.trg_emb_cat = VocabularyEmbedder(trg_voc_size, 2176)
        self.trg_cat = VocabularyEmbedder(trg_voc_size, 2176)           #2176为cat的尺寸
        self.pos_emb_subs  = PositionalEncoder(d_model_subs, dout_p)    #d_model_subs = 512 dout_p = 0.1
        self.pos_emb_imgs  = PositionalEncoder(d_model_imgs, dout_p)
        self.pos_emb_audio = PositionalEncoder(d_model_audio, dout_p)   #d_model_audio = 128
        self.pos_emb_video = PositionalEncoder(d_model_video, dout_p)   #d_model_video = 1024
        self.pos_emb_video_memory = PositionalEncoder(d_model_video, dout_p)  # d_model_video = 1024

        self.pos_emb_memory = PositionalEncoder(128, dout_p)  # d_model_video = 128
        # self.pos_emb_cat = PositionalEncoder(2176, dout_p)  # d_model_video = 1024
        self.encoder_subs  = Encoder(d_model_subs,  dout_p, H, d_ff_subs,  N_subs)
        self.encoder_imgs  = Encoder(d_model_imgs,  dout_p, H, d_ff_imgs,  N_subs)
        self.encoder_audio = Encoder(d_model_audio, dout_p, H, d_ff_audio, N_audio)
        self.encoder_video = Encoder(d_model_video, dout_p, H, d_ff_video, N_video)
        self.fusionLayer_video = Fusioner(d_model_video, dout_p, H, d_ff_video, N_video)
        self.encoder_video_memory = Encoder(d_model_video, dout_p, H, d_ff_video, N_video)                #d_ff_video = 2048      N_video = 1代表一层网络     H代表多头注意力的头数
        self.decoder_subs  = Decoder(d_model_subs,  dout_p, H, d_ff_subs,  N_subs)
        self.decoder_imgs  = Decoder(d_model_imgs,  dout_p, H, d_ff_imgs,  N_subs)
        self.decoder_audio = Decoder(d_model_audio, dout_p, H, d_ff_audio, N_audio)
        self.decoder_video = Decoder(d_model_video, dout_p, H, d_ff_video, N_video)
        self.decoder_video_memory = Decoder(128, dout_p, H, d_ff_video, N_video)
        self.decoder_cat   = Decoder(2176, dout_p, H, 2176 * 2, N_video)
        self.linear = nn.Linear(128, trg_voc_size)
        self.dropout = nn.Dropout(dout_p)
        self.linear2 = nn.Linear(trg_voc_size, trg_voc_size)
        self.memory_linear = nn.Linear(1024, 128)
        # late fusion
        self.generator = SubsAudioVideoGeneratorConcatLinearDoutLinear(
            d_model_subs, d_model_audio, d_model_video, trg_voc_size, dout_p
        )
        self.imgs_generator = ImgsSubsAudioVideoGeneratorConcatLinearDoutLinear(
            d_model_subs, d_model_imgs, d_model_audio, d_model_video, trg_voc_size, dout_p
        )
        self.regression_loss = torch.nn.MSELoss(reduction='none')  # none 不求平均 # 默认为mean #sum

        print('initialization: xavier')
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    # src_subs (B, Ss2, d_feat_subs), src_audio (B, Ss, d_feat_audio) src_video (B, Ss, d_feat_video) 
    # trg (B, St) src_mask (B, 1, Ss) src_sub_mask (B, 1, Ssubs) trg_mask (B, St, St)
    def forward(self, src, trg, mask):
        src_video, src_audio, src_subs, src_imgs = src
        src_mask, trg_mask, src_subs_mask, src_imgs_mask = mask
        # src_video_memory = src_video.clone()
        # print("trg", trg.shape)
        # print("src_subs",src_subs.shape,"src_video",src_video.shape, "src_audio",src_audio.shape)
        # print("src_subs_mask", src_subs_mask.shape, "trg_mask_mask", trg_mask.shape, "src_mask", src_mask.shape)
        # print("src_mask", src_mask.shape, "trg_mask", trg_mask.shape)       #src_mask torch.Size([64, 1, 225]) trg_mask torch.Size([64, 38, 38])
        # # embed
        # print("src_subs", src_subs.shape)          #src_subs torch.Size([32, 477， 512]) 477文字
        # print("src_imgs", src_imgs.shape)           #src_imgs torch.Size([32, 14，512]) 14文字
        # print("src_audio", src_audio.shape)         #src_audio torch.Size([32, 158, 128])
        # print("src_video", src_video.shape)         #src_video torch.Size([32, 158, 1024]) 158帧
        src_subs = self.src_emb_subs(src_subs)
        src_imgs = self.src_emb_imgs(src_imgs)
        src_audio = self.src_emb_audio(src_audio)
        src_video = self.src_emb_video(src_video)
        # src_video_memory = self.src_emb_video_memory(src_video_memory)

        # print("src_subs", src_subs.shape)
        # print("src_imgs", src_imgs.shape)
        # print("src_video",src_video.shape)      #src_video torch.Size([64, 225, 1024])
        trg_subs = self.trg_emb_subs(trg)
        trg_imgs = self.trg_emb_imgs(trg)
        trg_audio = self.trg_emb_audio(trg) ##寻找语义关系！！！！！！！！！！！！！！！！！！！！并且返回一个张量！！！！！！！！！！！！！！！
        trg_video = self.trg_emb_video(trg)
        # trg_video_memory = self.trg_emb_memory(trg)
        # print("trg", trg.shape)     #trg torch.Size([32, 37])
        # print("trg_subs", trg_subs.shape)           #trg_subs torch.Size([32, 37, 512])
        # print("trg_imgs", trg_imgs.shape)           #trg_imgs torch.Size([32, 37, 512])
        # print("trg_audio", trg_audio.shape)         #trg_audio torch.Size([32, 37, 128])
        # print("trg_video", trg_video.shape)         #trg_video torch.Size([32, 37, 1024])
        #





        src_subs = self.pos_emb_subs(src_subs)
        src_imgs = self.pos_emb_imgs(src_imgs)
        src_audio = self.pos_emb_audio(src_audio)
        src_video = self.pos_emb_video(src_video)
        # src_video_memory = self.pos_emb_video_memory(src_video_memory)
        # print("trg_subs", trg_subs.shape)
        # print("trg_imgs", trg_imgs.shape)
        # print("trg_video", trg_video.shape)             #trg_video torch.Size([64, 38, 1024])
        trg_subs = self.pos_emb_subs(trg_subs)          ##！！！！！！！！！！！！！！！！！！！！！！
        trg_imgs = self.pos_emb_imgs(trg_imgs)
        trg_audio = self.pos_emb_audio(trg_audio)
        trg_video = self.pos_emb_video(trg_video)
        # trg_video_memory = self.pos_emb_memory(trg_video_memory)
        # encode and decode
        # print("trg_subs", trg_subs.shape)             #trg_subs torch.Size([32, 41, 512])
        # print("trg_imgs", trg_imgs.shape)             #trg_imgs torch.Size([32, 41, 512])
        # print("trg_audio", trg_audio.shape)           #trg_audio torch.Size([32, 41, 128])
        # print("trg_video", trg_video.shape)           #trg_video torch.Size([32, 41, 1024])

        memory_subs = self.encoder_subs(src_subs, src_subs_mask)
        memory_imgs = self.encoder_imgs(src_imgs, src_imgs_mask)
        memory_video = self.encoder_video(src_video, src_mask)
        memory_audio = self.encoder_audio(src_audio, src_mask)
        memory_com = self.fusionLayer_video(memory_video, src_audio, src_mask)


        # memory_video_memory = self.encoder_video_memory(src_video_memory, src_mask)
        # print("memory_subs", memory_subs.shape)                     #memory_subs torch.Size([32, 396, 512])
        # print("memory_imgs", memory_imgs.shape)                     #memory_imgs torch.Size([32, 14, 512])
        # print("memory_audio", memory_audio.shape)                   #memory_audio torch.Size([32, 135, 128])
        # print("memory_video", memory_com.shape)                   #memory_video torch.Size([32, 135, 1024])


        ###联合loss训练器开始
        # masks = mask(feature_stacks[0][:, :, 0], trg, 1)
        # memory_video_audio = self.fusionLayer_video(src_video, src_audio, src_mask)
        # memory_video_audio = self.fusionLayer_video(memory_video, memory_audio, src_mask)
        # memory_video_memory = memory_video_memory.clone()
        # new_memory_video = self.memory_linear(F.relu(memory_video_memory))
        # print("new_memory_video", new_memory_video.shape)
        # print("trg_video", trg_video_memory.shape)
        # out_video_memory = self.decoder_video_memory(trg_video_memory, new_memory_video, src_mask, trg_mask)
        # out_video_memory = self.linear(out_video_memory)
        # out_video_memory = self.linear2(self.dropout(F.relu(out_video_memory)))
        # out_memory = F.log_softmax(out_video_memory, dim=-1)
        ###



        out_subs = self.decoder_subs(trg_subs, memory_subs, src_subs_mask, trg_mask)    #解码也要改！！！！！！！！！！！！！！！但是返回的还是一个张量！！！！！！！！！！
        out_imgs = self.decoder_imgs(trg_imgs, memory_imgs, src_imgs_mask, trg_mask)
        out_audio = self.decoder_audio(trg_audio, memory_audio, src_mask, trg_mask)
        out_video = self.decoder_video(trg_video, memory_com, src_mask, trg_mask)
        # print('out_subs.shape', out_subs.shape)
        # print('out_imgs.shape', out_imgs.shape)
        # print("out_video.shape", out_video.shape)
        # torch.con
        ###开始生成器
        # src_out_cat = torch.cat([out_subs, out_imgs, out_audio, out_video], dim=-1)
        # # print(src_out_cat.shape)
        # src_masks_out_cat, trg_mask_out_cat = mask_out(src_out_cat[:, :, 0], trg, 1)
        # src_out_cat = self.src_emb_video(src_out_cat)               #iden。。不做任何处理
        # src_out_cat = self.pos_emb_cat(src_out_cat)
        # trg_cat = self.trg_emb_cat(trg)
        # trg_cat = self.pos_emb_cat(trg_cat)
        # # memory_cat= self.encoder_cat(src_out_cat, src_masks_out_cat)
        # out_cat = self.decoder_cat(trg_cat, src_out_cat, src_masks_out_cat, trg_mask_out_cat)
        # x = self.linear(out_cat)
        # x = self.linear2(self.dropout(F.relu(x)))
        # # print("x2", x.shape)        #x2 torch.Size([64, 76, 10173])
        # return F.log_softmax(x, dim=-1)  # (B, seq_len, voc_size)
        ###结束

        ###联合loss
        # loss_sub = sim_matrix(out_subs, out_imgs)
        # loss = self.contrastive_loss(sim_matrix(out_subs, out_imgs))
        ###结束
        # print('out_video.shape', out_video.shape)                   #out_video.shape torch.Size([64, 38, 1024])
        # generate
        # src2_video = self.src2_emb_video(out_video)     #out_video  2048
        ##mse
        # loss = self.regression_loss(memory_audio, new_memory_video)
        # # loss = self.regression_loss(out_imgs, out_subs)
        # # print("regression_loss", loss.shape)
        # loss.requires_grad_(True)
        # loss.backward(torch.ones_like(loss), retain_graph=True)
        ###mse over
        out = self.imgs_generator(out_subs, out_imgs, out_audio, out_video)    #返回的还是一个张量!!!!!!!!!!!!
        # print("out",out.shape)      #torch.Size([64, 38, 10173])

        return out # (B, St, voc_size)
