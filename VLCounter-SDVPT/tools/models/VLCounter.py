import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ViT_Encoder import VPTCLIPVisionTransformer as vpt
from .ViT_Encoder_add import SPTCLIPVisionTransformer as spt
from .Text_Encoder import CLIPTextEncoder

from timm.models.layers import trunc_normal_

def trunc_normal_init(module: nn.Module,
                      mean: float = 0,
                      std: float = 1,
                      a: float = -2,
                      b: float = 2,
                      bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)  # type: ignore


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding=0, flag=True):
        super(UpConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, padding=padding)
        if flag:
            self.gn = nn.GroupNorm(8, out_channels)
        else:
            self.gn = nn.GroupNorm(1, out_channels)
        self.gelu = nn.GELU()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.flag = flag

    def forward(self, trg):
        trg = self.conv(trg)
        if self.flag:
            trg = self.up(self.gelu(self.gn(trg)))
        return trg


class Counter(nn.Module):
    def __init__(self, args):
        super(Counter,self).__init__()

        self.v = args.v
        self.enc = args.enc

        embed_dims = 512
        proj_dims = 64
        self.t_proj = nn.Linear(embed_dims, proj_dims)
        self.v_proj = nn.Linear(embed_dims, proj_dims)

        self.proj = nn.Sequential(
            nn.Conv2d(768, proj_dims, 1),
            nn.GroupNorm(8, proj_dims),
            nn.GELU(),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self.proj1 = nn.Sequential(
            nn.Conv2d(768, proj_dims, 1),
            nn.GroupNorm(8, proj_dims),
            nn.GELU(),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )
        self.proj2 = nn.Sequential(
            nn.Conv2d(768, proj_dims, 1),
            nn.GroupNorm(8, proj_dims),
            nn.GELU(),
            nn.UpsamplingBilinear2d(scale_factor=8)
        )
        self.decoder = nn.ModuleList([
                                    UpConv(proj_dims+1, proj_dims, 3, 1),
                                    UpConv(proj_dims, proj_dims, 3,1),
                                    UpConv(proj_dims, proj_dims, 3, 1),
                                    UpConv(proj_dims, proj_dims, 3,1),
                                    UpConv(proj_dims, 1, 1, flag=False)
                                ])


        self.attn_weight = nn.Parameter(torch.ones(1, 1, 24, 24))
        self.attn_bias = nn.Parameter(torch.zeros(1, 1, 24, 24))
        self.init_weights()

        if args.enc == "spt":
            self.v_enc = spt(pretrained=args.MODEL.pretrain+'ViT-B-16.pt', num_tokens=args.num_tokens, patch_size=args.patch_size)
            self.v_enc.init_weights()
        elif args.enc == "vpt":
            self.v_enc = vpt(pretrained=args.MODEL.pretrain+'ViT-B-16.pt')
            self.v_enc.init_weights()
        else:
            raise NotImplementedError
        
        self.t_enc = CLIPTextEncoder(pretrained=args.MODEL.pretrain+'ViT-B-16.pt', embed_dim=embed_dims)
        self.t_enc.init_weights()

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, v, tokenized_text,stage, class_ids,_t_train):
        B = v.size(0)

        t = []

        for index,tt in enumerate(tokenized_text):
            _t = self.t_enc(tt)
            _t = _t / _t.norm(dim=-1, keepdim=True)
            _t = _t.mean(dim=0)
            _t /= _t.norm()
            t.append(_t)
 

        _t = torch.stack(t)

        if stage=='train':
            sim_list= None
            class_index_list = None
        else:
            sim = F.cosine_similarity(_t.unsqueeze(1), _t_train.unsqueeze(0), dim=-1)

            #apply self-mask
            new_sim = sim.detach().clone()
            if stage == 'train2': 
                class_ids_tensor = torch.tensor(class_ids, dtype=torch.long, device=sim.device)
                min_new_sim = new_sim.min(dim=1, keepdim=True).values  
                sim.scatter_(1, class_ids_tensor.unsqueeze(1), min_new_sim)

            # normalize
            sim_min = sim.min(dim=1, keepdim=True).values  # [32, 1]
            sim_max = sim.max(dim=1, keepdim=True).values  # [32, 1]
            sim_normalized = (sim - sim_min) / (sim_max - sim_min)  # [32, 89]
            sim_normalized, class_index_list = torch.topk(sim_normalized, 70)
            sim_sum = sim_normalized.sum(dim=1, keepdim=True)  # [32, 1]
            sim_list = sim_normalized / sim_sum  # [32, 89]
  
        if self.enc == "vpt":
            v,outs_vpt_class,outs_vpt_weighted = self.v_enc(v,stage,class_ids,sim_list,class_index_list)
        elif self.enc == "spt":
            v,outs_vpt_class,outs_vpt_weighted = self.v_enc(v, _t.unsqueeze(1),stage,class_ids,sim_list,class_index_list)
        else:
            raise NotImplementedError

        proj_v, _t = self.d3_to_d4(self.v_proj(self.d4_to_d3(v[-1]))), self.t_proj(_t)
        attn_map = torch.einsum('bc,bchw->bhw', _t, proj_v).unsqueeze(1)
        affine_attn_map = self.attn_weight.expand(B, -1, -1, -1) * attn_map + self.attn_bias.expand(B, -1, -1, -1)
        
        x = torch.cat([proj_v, affine_attn_map], dim=1)
        for i, d in enumerate(self.decoder):
            if i==1:
                x = d(x + self.proj(v[-2]) * F.interpolate(affine_attn_map, scale_factor=2))
            elif i==2:
                x = d(x + self.proj1(v[-3]) * F.interpolate(affine_attn_map, scale_factor=4))
            elif i==3:
                x = d(x + self.proj2(v[-4]) * F.interpolate(affine_attn_map, scale_factor=8))
            else:
                x = d(x)
        return x, F.interpolate(affine_attn_map, scale_factor=16), affine_attn_map,outs_vpt_class,outs_vpt_weighted 

    def d3_to_d4(self, t):
        b, hw, c = t.size()
        if hw % 2 != 0:
            t = t[:, 1:]
        h = w = int(math.sqrt(hw))
        return t.transpose(1, 2).reshape(b, c, h, w)

    def d4_to_d3(self, t):
        return t.flatten(-2).transpose(-1, -2)
