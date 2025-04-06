
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# Useful debugging function to check if any value is NaN during training
def check_nan(tensor, name):
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN in {name}")
        
class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

    

class CrossAttention(nn.Module):
    def __init__(self, channels, context_channels, num_heads=4):
        super().__init__()
        self.context_norm = nn.GroupNorm(1, context_channels)
        self.mha = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5  

        self.context_proj = nn.Sequential(
            nn.Conv2d(context_channels, 2*channels, kernel_size=1),
            nn.GroupNorm(1, 2*channels)  # Equivalent to LayerNorm for channels

        )

    def forward(self, x, context):
        context = self.context_norm(context) 
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H*W).permute(0, 2, 1)  # (B, H*W, C)
        x_ln = self.ln(x_flat)
        
        # Project context to keys and values
        kv = self.context_proj(context)  # (B, 2*C, H_ctx, W_ctx)
        k, v = kv.chunk(2, dim=1)  # (B, C, H_ctx, W_ctx) each
        
        k = k.view(B, C, -1).permute(0, 2, 1)  # (B, H_ctx*W_ctx, C)
        v = v.view(B, C, -1).permute(0, 2, 1)  # (B, H_ctx*W_ctx, C)
        
        attn_output, _ = self.mha(x_ln, k, v)
        attn_output = attn_output * self.scale  
        attn_output = attn_output + x_flat
        attn_output = attn_output.permute(0, 2, 1).view(B, C, H, W)
        return attn_output
    
class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )
    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, image_size=64, c_in=2, c_out=2, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.image_size = image_size
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, image_size//2)
        self.down2 = Down(128, 196)
        self.sa2 = SelfAttention(196, image_size//4)
        self.down3 = Down(196, 196)
        self.sa3 = SelfAttention(196, image_size//8)

        self.bot1 = DoubleConv(196, 392)
        self.bot2 = DoubleConv(392, 392)
        self.bot3 = DoubleConv(392, 196)

        # x = self.up2(x, x2, t)

        self.up1 = Up(392, 128)
        self.sa4 = SelfAttention(128, image_size//4)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, image_size//2)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, image_size)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output
    
class UNet_Mag_Only(nn.Module):
    def __init__(self, image_size=64, c_in=1, c_out=1, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.image_size = image_size
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, image_size//2)
        self.down2 = Down(128, 196)
        self.sa2 = SelfAttention(196, image_size//4)
        self.down3 = Down(196, 196)
        self.sa3 = SelfAttention(196, image_size//8)

        self.bot1 = DoubleConv(196, 392)
        self.bot2 = DoubleConv(392, 392)
        self.bot3 = DoubleConv(392, 196)

        # x = self.up2(x, x2, t)

        self.up1 = Up(392, 128)
        self.sa4 = SelfAttention(128, image_size//4)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, image_size//2)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, image_size)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


class UNet_conditional(nn.Module):
    def __init__(self, image_size=64, c_in=6, c_out=6, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.image_size = image_size
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, image_size//2)
        self.down2 = Down(128, 196)
        self.sa2 = SelfAttention(196, image_size//4)
        self.down3 = Down(196, 196)
        self.sa3 = SelfAttention(196, image_size//8)

        self.bot1 = DoubleConv(196, 392)
        self.bot2 = DoubleConv(392, 392)
        self.bot3 = DoubleConv(392, 196)

        # x = self.up2(x, x2, t)

        self.up1 = Up(392, 128)
        self.sa4 = SelfAttention(128, image_size//4)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, image_size//2)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, image_size)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        self.mag_and_phase_encoder = nn.Sequential(
            nn.Conv2d(c_in, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(),
            nn.Linear(256, time_dim) 
        )


    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, rain_fft):
        """
        Inputs:
        x : an BxWxHx6 array, where the first BxNx3 is the magnitude component of the 
            difference, and the last BxNx3 is the phase component of the difference
        t : The timestep, higher t = more noise 
        rain_fft : The condition, which is an BxWxHx6 array, and is obtained by concatenating
                   the phase and magnitude components of the rained image, same to the structure of x
        """
        

        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if rain_fft is not None:
            t += self.mag_and_phase_encoder(rain_fft)


        x1 = self.inc(x)
        x2 = self.down1(x1, t)

        x2 = self.sa1(x2)

        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)

        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output
    
class UNet_conditional_YCrCb(nn.Module):
    def __init__(self, image_size=64, c_in=4, c_out=2, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.image_size = image_size
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, image_size//2)
        self.down2 = Down(128, 196)
        self.sa2 = SelfAttention(196, image_size//4)
        self.down3 = Down(196, 196)
        self.sa3 = SelfAttention(196, image_size//8)

        self.bot1 = DoubleConv(196, 392)
        self.bot2 = DoubleConv(392, 392)
        self.bot3 = DoubleConv(392, 196)

        # x = self.up2(x, x2, t)

        self.up1 = Up(392, 128)
        self.sa4 = SelfAttention(128, image_size//4)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, image_size//2)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, image_size)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        self.mag_and_phase_encoder = nn.Sequential(
            nn.Conv2d(c_in, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(),
            nn.Linear(256, time_dim) 
        )


    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, rain_fft):
        """
        Inputs:
        x : an BxWxHx2 array, where the first BxNx1 is the magnitude component of the 
            difference in the Y channel, and the last BxNx1 is the phase component of the difference
            in the Y channel as well
        t : The timestep, higher t = more noise 
        rain_fft : The condition, which is an BxWxHx2 array, and is obtained by concatenating
                   the phase and magnitude components of the rained image in the Y channel, 
                   same to the structure of x
        """
        

        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if rain_fft is not None:
            x = torch.cat([x, rain_fft], dim=1)
        else:
            zero_cond = torch.zeros_like(x)
            x = torch.cat([x, zero_cond], dim=1)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)

        x2 = self.sa1(x2)

        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)

        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output
    
# YCrCb UNet but with cross attention
class UNet_conditional_YCrCb_CA(nn.Module):
    def __init__(self, image_size=64, c_in=2, c_out=2, time_dim=256, device="cuda"):
        self.c_in = c_in
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.image_size = image_size
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, image_size//2)
        self.ca1 = CrossAttention(128, context_channels=256)

        self.down2 = Down(128, 196)
        self.sa2 = SelfAttention(196, image_size//4)
        self.ca2 = CrossAttention(196, context_channels=256)

        self.down3 = Down(196, 196)
        self.sa3 = SelfAttention(196, image_size//8)
        self.ca3 = CrossAttention(196, context_channels=256)

        self.bot1 = DoubleConv(196, 392)
        self.bot2 = DoubleConv(392, 392)
        self.bot3 = DoubleConv(392, 196)

        # x = self.up2(x, x2, t)

        self.up1 = Up(392, 128)
        self.sa4 = SelfAttention(128, image_size//4)
        self.ca4 = CrossAttention(128, context_channels=256)

        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, image_size//2)
        self.ca5 = CrossAttention(64, context_channels=256)

        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, image_size)
        self.ca6 = CrossAttention(64, context_channels=256)

        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        # self.mag_and_phase_encoder = nn.Sequential(
        #     nn.Conv2d(c_in, 64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool2d((1, 1)), 
        #     nn.Flatten(),
        #     nn.Linear(256, time_dim) 
        # )

        self.mag_and_phase_encoder = nn.Sequential(
            nn.Conv2d(c_in, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )


    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, rain_fft):
        """
        Inputs:
        x : an BxWxHx2 array, where the first BxNx1 is the magnitude component of the 
            difference in the Y channel, and the last BxNx1 is the phase component of the difference
            in the Y channel as well
        t : The timestep, higher t = more noise 
        rain_fft : The condition, which is an BxWxHx2 array, and is obtained by concatenating
                   the phase and magnitude components of the rained image in the Y channel, 
                   same to the structure of x
        """
        

        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        if rain_fft is None:
            rain_fft = torch.zeros(x.size(0), self.c_in, self.image_size, self.image_size).to(x.device)
            
        context = self.mag_and_phase_encoder(rain_fft)

        x1 = self.inc(x)

        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x2 = self.ca1(x2, context)

        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x3 = self.ca2(x3, context)

        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)
        x4 = self.ca3(x4, context)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.ca4(x, context)

        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.ca5(x, context)


        x = self.up3(x, x1, t)
        x = self.sa6(x)
        x = self.ca6(x, context)
        
        output = self.outc(x)
        return output


class UNet_MNIST(nn.Module):

    def __init__(self, channels=[32, 64, 128, 256], embed_dim=256):
        """
        Initialize a time-dependent unet.

        Args:
        channels: The number of channels for feature maps of each resolution.
        embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim))
        self.y_embed = nn.Embedding(10, embed_dim)
        # B x 28 x 28 
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=channels[0], kernel_size=3)
        self.gnorm1 = nn.GroupNorm(num_groups=4, num_channels=channels[0])
        self.dense1 = Dense(embed_dim, channels[0])  

        # B x 12 x 12 x 32 (?) 
        self.conv2 = nn.Conv2d(in_channels=channels[0] , out_channels=channels[1], kernel_size=3, padding = 1, stride=2)
        self.gnorm2 = nn.GroupNorm(num_groups=32, num_channels=channels[1])
        self.dense2 = Dense(embed_dim, channels[1])  

        self.conv3 = nn.Conv2d(in_channels=channels[1] , out_channels=channels[2], kernel_size=3, padding = 1, stride=2)
        self.gnorm3 = nn.GroupNorm(num_groups=64, num_channels=channels[2])
        self.dense3 = Dense(embed_dim, channels[2])  

        self.conv4 = nn.Conv2d(in_channels=channels[2] , out_channels=channels[3], kernel_size=3, padding = 1, stride=2)
        self.gnorm4 = nn.GroupNorm(num_groups=128, num_channels=channels[3])
        self.dense4 = Dense(embed_dim, channels[3])  

        self.attn3 = SelfAttention(channels[2]) 
        self.attn4 = SelfAttention(channels[3]) 

        self.tconv4 = nn.ConvTranspose2d(in_channels=channels[3], out_channels=channels[2], kernel_size=3, stride=2, padding=1)
        self.tgnorm4 = nn.GroupNorm(num_groups=64, num_channels=channels[2])
        self.tdense4 = Dense(embed_dim, channels[2])  

        self.tconv3 = nn.ConvTranspose2d(in_channels=channels[2], out_channels=channels[1], kernel_size=3, stride=2, padding=1)
        self.tgnorm3 = nn.GroupNorm(num_groups=32, num_channels=channels[1])
        self.tdense3 = Dense(embed_dim, channels[1])  

        self.tconv2 = nn.ConvTranspose2d(in_channels=channels[1], out_channels=channels[0], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.tgnorm2 = nn.GroupNorm(num_groups=4, num_channels=channels[0])
        self.tdense2 = Dense(embed_dim, channels[0])  

        self.tconv1 = nn.ConvTranspose2d(in_channels=channels[0], out_channels=1, kernel_size=3)
        # The swish activation function

        self.act = nn.SiLU()
    

    def forward(self, x, t, cond=None):
        """
        Overall:

        input -> downsample + add timeembedding + normalize + activation -> repeat 4 times
        Also, add 2 self-attention block around the bottleneck 
        -> Reverse is also the same except we upsample instead of downsample
        """

        x = self.conv1(x)
        ori_t = self.embed(t)
        if cond is not None:
            comb_embed = self.y_embed(cond)
            ori_t += comb_embed 

        # NOTE: if doing this with Rain FFT image cross attention 
        #       might be more helpful than just adding
        time_embedding = self.dense1(ori_t).unsqueeze(-1)
        x = x + time_embedding
        x = self.gnorm1(x)
        x = self.act(x)
        x1 = x

        x = self.conv2(x)
        time_embedding = self.dense2(ori_t).unsqueeze(-1)
        x = x + time_embedding
        x = self.gnorm2(x)
        x = self.act(x)
        x2 = x

        x = self.conv3(x)
        x = self.attn3(x)
        time_embedding = self.dense3(ori_t).unsqueeze(-1)
        x = x + time_embedding
        x = self.gnorm3(x)
        x = self.act(x)
        x3 = x

        
        x = self.conv4(x)
        x = self.attn4(x)
        time_embedding = self.dense4(ori_t).unsqueeze(-1)
        x = x + time_embedding
        x = self.gnorm4(x)
        x = self.act(x)

        # Upsampling 
        x = self.tconv4(x) 

        time_embedding = self.tdense4(ori_t).unsqueeze(-1)
        x = x + time_embedding
        x = self.tgnorm4(x)
        x = self.act(x)
        x = x + x3

        x = self.tconv3(x) 
        time_embedding = self.tdense3(ori_t).unsqueeze(-1)
        x = x + time_embedding
        x = self.tgnorm3(x)
        x = self.act(x)
        x = x + x2

        x = self.tconv2(x) 
        time_embedding = self.tdense2(ori_t).unsqueeze(-1)
        x = x + time_embedding
        x = self.tgnorm2(x)
        x = self.act(x)
        x = x + x1
        x = self.tconv1(x)

        return x
    
if __name__ == '__main__':
    # net = UNet(device="cpu")
    net = UNet_conditional(num_classes=10, device="cpu")
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(3, 3, 64, 64)
    t = x.new_tensor([500] * x.shape[0]).long()
    y = x.new_tensor([1] * x.shape[0]).long()
    print(net(x, t, y).shape)