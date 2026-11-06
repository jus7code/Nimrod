"""
Autoencoder con encoder compartido y dos decoders (estilo DeepFaceLab "DF").

            ┌─► Decoder_A ─► reconstruye caras de A (fuente)
  Encoder ──┤
            └─► Decoder_B ─► reconstruye caras de B (video)

  Swap: cara de B ─► Encoder ─► Decoder_A ─► A con la pose/expresión/luz de B

El cuello de botella denso (fc1 → 512 → fc2) obliga al encoder a guardar solo lo que
comparten ambas personas (pose, expresión, iluminación) y no los detalles de identidad,
que quedan "dibujados" en los pesos de cada decoder.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

RES = 128


class Down(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(nn.Conv2d(cin, cout, 5, 2, 2), nn.LeakyReLU(0.1, inplace=True))


class Up(nn.Sequential):
    """Upsampling ×2 con PixelShuffle (evita los artefactos de tablero de ConvTranspose)."""

    def __init__(self, cin: int, cout: int):
        super().__init__(nn.Conv2d(cin, cout * 4, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True), nn.PixelShuffle(2))


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1)

    def forward(self, x):
        y = F.leaky_relu(self.c1(x), 0.2)
        return F.leaky_relu(x + self.c2(y), 0.2)


class Encoder(nn.Module):
    def __init__(self, ch: int = 64, latent: int = 512):
        super().__init__()
        self.down = nn.Sequential(Down(3, ch), Down(ch, ch * 2), Down(ch * 2, ch * 4), Down(ch * 4, ch * 8))
        s = RES // 16                       # 128 -> 8×8
        self.shape = (ch * 8, s, s)
        self.fc1 = nn.Linear(ch * 8 * s * s, latent)
        self.fc2 = nn.Linear(latent, ch * 8 * s * s)
        self.up = Up(ch * 8, ch * 8)       # 8 -> 16

    def forward(self, x):
        x = self.down(x).flatten(1)
        x = self.fc2(self.fc1(x)).view(-1, *self.shape)
        return self.up(x)                    # (B, 512, 16, 16)


class Decoder(nn.Module):
    def __init__(self, ch: int = 64):
        super().__init__()
        # Sin bloque residual a 128 px: era el ~15 % del cómputo y casi no aporta calidad.
        self.net = nn.Sequential(
            Up(ch * 8, ch * 4), ResBlock(ch * 4),   # 16 -> 32
            Up(ch * 4, ch * 2), ResBlock(ch * 2),   # 32 -> 64
            Up(ch * 2, ch),                         # 64 -> 128
            nn.Conv2d(ch, 3, 5, 1, 2),
        )

    def forward(self, z):
        return torch.sigmoid(self.net(z))


class SwapAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder_a = Decoder()
        self.decoder_b = Decoder()

    def swap_to_a(self, x):
        return self.decoder_a(self.encoder(x))


def dssim(x: torch.Tensor, y: torch.Tensor, ks: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """(1 - SSIM) / 2: penaliza diferencias de estructura/contraste local, no solo de píxel."""
    c = x.shape[1]
    g = torch.arange(ks, device=x.device, dtype=x.dtype) - ks // 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())
    w = (g[:, None] * g[None, :]).expand(c, 1, ks, ks).contiguous()
    blur = lambda t: F.conv2d(t, w, padding=ks // 2, groups=c)  # noqa: E731
    mx, my = blur(x), blur(y)
    sxx = blur(x * x) - mx ** 2
    syy = blur(y * y) - my ** 2
    sxy = blur(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
    return ((1 - ssim) / 2).mean()
