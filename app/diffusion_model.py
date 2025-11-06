# app/diffusion_model.py
import torch
import torch.nn as nn

# Minimal UNet-ish backbone for DDPM on 32x32
def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.GroupNorm(8, cout),
        nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.GroupNorm(8, cout),
        nn.SiLU(),
    )

class TinyUNet(nn.Module):
    def __init__(self, in_ch=3, base=64):
        super().__init__()
        self.down1 = conv_block(in_ch, base)
        self.down2 = conv_block(base, base*2)
        self.down3 = conv_block(base*2, base*4)
        self.pool = nn.AvgPool2d(2)
        self.mid  = conv_block(base*4, base*4)
        self.up3  = conv_block(base*8, base*2)
        self.up2  = conv_block(base*4, base)
        self.up1  = nn.Sequential(
            nn.Conv2d(base*2, base, 3, padding=1), nn.SiLU(),
            nn.Conv2d(base, in_ch, 3, padding=1),
        )

    def forward(self, x, t_embed):
        # (optional) add t_embed as FiLM/bias; keep minimal for clarity
        d1 = self.down1(x)              # 32x32
        d2 = self.down2(self.pool(d1))  # 16x16
        d3 = self.down3(self.pool(d2))  # 8x8
        m  = self.mid(self.pool(d3))    # 4x4
        u3 = torch.nn.functional.interpolate(m, scale_factor=2, mode="nearest")
        u3 = self.up3(torch.cat([u3, d3], dim=1))
        u2 = torch.nn.functional.interpolate(u3, scale_factor=2, mode="nearest")
        u2 = self.up2(torch.cat([u2, d2], dim=1))
        u1 = torch.nn.functional.interpolate(u2, scale_factor=2, mode="nearest")
        out= self.up1(torch.cat([u1, d1], dim=1))
        return out  # predict noise ε
