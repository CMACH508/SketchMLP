import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Hyper_params import hp
from einops import rearrange
from ASMLP.as_mlp import AxialShiftedBlock
from torch import einsum

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Attention(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner, causal=False):
        super().__init__()
        self.scale = dim_inner ** -0.5
        self.causal = causal

        self.to_qkv = nn.Linear(dim_in, dim_inner * 3, bias=False)
        self.to_out = nn.Linear(dim_inner, dim_out)

    def forward(self, x):
        device = x.device
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # 还是用全连接层做q,k,v
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # causal 取false
        if self.causal:
            mask = torch.ones(sim.shape[-2:], device=device).triu(1).bool()
            sim.masked_fill_(mask[None, ...], -torch.finfo(q.dtype).max)

        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        return self.to_out(out)
    # softmax(k*q)*v


class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len):
        super(SpatialGatingUnit, self).__init__()
        self.norm = nn.LayerNorm(d_ffn//2)
        self.spatial_proj = nn.Conv1d(seq_len, seq_len, 1)
        #self.attn = Attention(d_ffn, d_ffn//2, 64)

    def forward(self, x):
        u,v = torch.chunk(x, 2, dim=-1)
        v = self.norm(v)
        v = self.spatial_proj(v) #+ self.attn(x)
        out = u*v
        return out


class GMLPblock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len, dpr=0.0):
        super(GMLPblock, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn)
        self.channel_proj2 = nn.Linear(d_ffn//2, d_model)
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)

        self.droppath = DropPath(dpr) if dpr > 0.0 else nn.Identity()

    def forward(self, x):
        residual = self.norm(x)
        x = F.gelu(self.channel_proj1(x))
        x = self.sgu(x)
        x = self.channel_proj2(x)
        out = self.droppath(x)+residual
        return out


class Com_unit(nn.Module):
    def __init__(self, in_ch, out_ch, d_in, d_out):
        super(Com_unit, self).__init__()
        self.norm = nn.LayerNorm(d_in)
        self.channel_proj1 = nn.Linear(d_in, d_out)
        self.spatial_proj = nn.Conv1d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.norm(x)
        x = F.gelu(self.channel_proj1(x))
        x = F.gelu(self.spatial_proj(x))
        return x


class Com_block(nn.Module):
    def __init__(self):
        super(Com_block, self).__init__()
        d_model = hp.d_model
        d_ffn = hp.d_ffn
        c_com = hp.c_com

        self.num_patches = hp.num_patches
        self.i2e = Com_unit(self.num_patches, c_com, d_model, d_model)
        self.s2e = Com_unit(hp.seq_len, c_com, d_model, d_model)
        self.att_proj = nn.Sequential(nn.Linear(c_com*2, self.num_patches+hp.seq_len), nn.Sigmoid())

    def forward(self, img_emb, seq_emb):
        img_pool = torch.mean(self.i2e(img_emb), -1)
        seq_pool = torch.mean(self.s2e(seq_emb), -1)
        att_emb = torch.cat([img_pool,seq_pool], dim=-1)
        attention = self.att_proj(att_emb)
        img_att = attention[:, :self.num_patches].unsqueeze(2).repeat(1,1,hp.d_model)
        seq_att = attention[:, self.num_patches:].unsqueeze(2).repeat(1,1,hp.d_model)
        seq_out = seq_emb * seq_att
        img_out = img_emb * img_att
        return img_out, seq_out


class img_head(nn.Module):
    def __init__(self):
        super(img_head, self).__init__()
        self.num_patches = hp.num_patches
        self.patcher = nn.Sequential(nn.Conv2d(hp.img_ch, 48, 7, stride=4, padding=3), nn.BatchNorm2d(48), nn.GELU(),
                                     nn.Conv2d(48, hp.d_model, 7, stride=4, padding=3))

        #self.patcher = nn.Sequential(nn.Conv2d(hp.img_ch, 32, 7, stride=4, padding=3), nn.BatchNorm2d(32), nn.GELU(),
                                     #nn.Conv2d(32, hp.d_model, 7, stride=4, padding=3))
        self.layers = nn.ModuleList(
            [AxialShiftedBlock(hp.d_model, hp.img_size, drop_path=hp.drop_rate) for i in range(hp.net_struct[0])])
        # self.layers = nn.ModuleList([CycleBlock(hp.d_model, hp.d_ffn, drop_path=hp.drop_rate) for i in range(hp.net_struct[0])])


    def forward(self, img):
        x = self.patcher(img)
        # B,C,H,W-> B,H,W,C
        x = x.permute(0, 2, 3, 1)
        for layer in self.layers:
            x = layer(x)
        B, H, W, C = x.shape
        x = x.reshape((B, -1, C))
        out = x
        return out


class seq_head(nn.Module):
    def __init__(self):
        super(seq_head, self).__init__()
        d_model = hp.d_model
        d_ffn = hp.d_ffn
        self.emb = nn.Sequential(nn.Linear(hp.sf_num, d_model))
        self.layers = nn.ModuleList([GMLPblock(d_model, d_ffn, 100, dpr=hp.drop_rate) for i in range(hp.net_struct[0])])

    def forward(self, seq):
        B = seq.shape[0]
        #cls_flag = torch.zeros(B,1,hp.d_model).cuda()
        x = self.emb(seq)[:,:100,:]
        #x = torch.cat([cls_flag,x],dim=1)
        for layer in self.layers:
            x = layer(x)
        out = x
        return out


class net_block(nn.Module):
    def __init__(self, bid, in_ch):
        super(net_block, self).__init__()
        d_model = hp.d_model
        d_ffn = hp.d_ffn
        #self.layers = nn.ModuleList([GMLPblock(d_model, d_ffn, in_ch, dpr=hp.drop_rate) for i in range(1)])
        self.layers = nn.ModuleList([GMLPblock(d_model, d_ffn, in_ch, dpr=hp.drop_rate) for i in range(hp.net_struct[bid])])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        out = x
        return out


class img_branch_block(nn.Module):
    def __init__(self, bid, in_ch):
        super(img_branch_block, self).__init__()
        self.layers = nn.ModuleList(
            [AxialShiftedBlock(hp.d_model, hp.img_size, drop_path=hp.drop_rate) for i in range(hp.net_struct[bid])])

    def forward(self, x):
        # B L C -> B H W C
        x = x.view(x.shape[0], hp.img_size//hp.patch_size, hp.img_size//hp.patch_size, hp.d_model)
        for layer in self.layers:
            x = layer(x)
        B, H, W, C = x.shape
        x = x.reshape((B, -1, C))
        out = x
        return out

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.img_head = img_head()
        self.seq_head = seq_head()
        self.img_b1 = img_branch_block(1, hp.num_patches)
        self.seq_b1 = net_block(1, 100)
        self.img_b2 = img_branch_block(2, hp.num_patches)
        self.seq_b2 = net_block(2, 100)
        self.img_b3 = img_branch_block(3, hp.num_patches)
        self.seq_b3 = net_block(3, 100)
        self.img_b4 = img_branch_block(4, hp.num_patches)
        self.seq_b4 = net_block(4, 100)


        self.img_classifier = nn.Sequential(nn.LayerNorm(hp.d_model),
                                            nn.Linear(hp.d_model, hp.categories))
        self.seq_classifier = nn.Sequential(nn.LayerNorm(hp.d_model),
                                            nn.Linear(hp.d_model, hp.categories))
        # self.expert = nn.Sequential(nn.LayerNorm(hp.d_model*2), nn.Linear(hp.d_model*2, 512),nn.GELU(), nn.Dropout(hp.drop_rate),
        #                             nn.Linear(512, 2), nn.Softmax())
        self.expert = nn.Sequential(nn.LayerNorm(hp.d_model*2), nn.Linear(hp.d_model*2, 512),nn.GELU(), nn.Dropout(hp.drop_rate),
                                    nn.Linear(512, 2))

    def forward(self, img, seq):
        B = seq.shape[0]
        seq_emb = self.seq_head(seq)
        img_emb = self.img_head(img)

        seq_emb = self.seq_b1(seq_emb)
        img_emb = self.img_b1(img_emb)
 
        seq_emb = self.seq_b2(seq_emb)
        img_emb = self.img_b2(img_emb)

        seq_emb = self.seq_b3(seq_emb)
        img_emb = self.img_b3(img_emb)

        seq_emb = self.seq_b4(seq_emb)
        img_emb = self.img_b4(img_emb)

        seq_emb = torch.mean(seq_emb, dim=1).view(B, -1)
        img_emb = torch.mean(img_emb, dim=1).view(B, -1)
        img_ans = self.img_classifier(img_emb)
        seq_ans = self.seq_classifier(seq_emb)

        expert = self.expert(torch.cat([img_emb,seq_emb],dim=1))
        img_jdg = expert[:,0].unsqueeze(1).repeat(1,hp.categories)
        seq_jdg = expert[:,1].unsqueeze(1).repeat(1,hp.categories)
        out = img_jdg*img_ans + seq_jdg*seq_ans
        return out, img_ans, seq_ans, img_jdg

    def get_auxloss(self, img_jdg):
        return ((img_jdg-0.5)**2)


