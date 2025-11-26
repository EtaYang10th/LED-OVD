from .backbone import build_backbone

import torch.nn as nn
import torch
import re
import ipdb
import torch.nn.functional as F



class Projector(nn.Module):
    def __init__(
        self,
        args,
        mm_hidden_size,
        hidden_size,
        projector_type,
        scale_factor=0.25,
    ):
        super().__init__()
        
        self.scale_factor = scale_factor
        self.backbone  = build_backbone(args)

        self.align_type = args.align_type

        if self.align_type == 'use_mlp': 
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(mm_hidden_size*int(1/self.scale_factor)**2, hidden_size)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU())
                    modules.append(nn.Linear(hidden_size, hidden_size))
                self.projector = nn.Sequential(*modules)
            else:
                raise ValueError(f"Unknown projector type {projector_type}")
        if self.align_type == 'repeat_crop':
            self.mlp1 = nn.Sequential(
            nn.LayerNorm(4096),
            nn.Linear(4096, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def pixel_shuffle(self, x, scale_factor):
        n, w, h, c = x.shape
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.reshape(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.reshape(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))

        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def resize_hw(self, h, w, targe):
        if ((h ) % targe) != 0:
            h = ((h ) // targe + 1) * targe 
        if ((w ) % targe) != 0:
            w = ((w ) // targe + 1) * targe
        return h, w
    

    def forward(self, samples):
        with torch.no_grad():
            features, position_embedding_swin = self.backbone(samples)
        visual_features = features[0].decompose()[0]

        b,c,h,w = visual_features.size()
        h, w = self.resize_hw(h, w, 4)
        visual_features = F.interpolate(visual_features, size=(h, w), mode='bilinear').to(visual_features.device)
        # [B, C, H, W] -> [B, H, W, C]
        visual_features = visual_features.permute(0, 2, 3, 1)

        visual_features = self.pixel_shuffle(visual_features, scale_factor=self.scale_factor)
        bs, c, h, w = visual_features.size()

        if self.align_type == 'use_mlp':
            visual_features = self.projector(visual_features)
        elif self.align_type == 'repeat_crop':
            # 将特征图在通道维度上重复一次，再截断到4096维
            visual_features = torch.cat([visual_features, visual_features], dim=-1)
            visual_features = visual_features[:, :, :, :4096]
            visual_features = self.mlp1(visual_features)
        else:
            raise ValueError(f"Unknown align type {self.align_type}")
        
        return visual_features
    

def build_projector(args, config):
    return Projector(
                    args,
                    mm_hidden_size=192,
                    hidden_size=896,
                    projector_type = 'mlp2x_gelu',
                    scale_factor=0.25,
                    )