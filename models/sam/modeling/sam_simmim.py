import torch
import torch.nn as nn
import torch.nn.functional as F


class SamSimMIM(nn.Module):
    def __init__(
        self,
        image_encoder,
        embed_dim: int,
        feat_dim: int = 256,
        patch_size: int = 16,
        in_chans: int = 4,
    ):
        super().__init__()

        self.image_encoder = image_encoder
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.lambda_nir = 0.5

        self.mask_token_rgb = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.mask_token_nir = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token_rgb, std=0.02)
        nn.init.trunc_normal_(self.mask_token_nir, std=0.02)

        self.decoder = nn.Sequential(
            nn.Conv2d(
                feat_dim,
                (patch_size ** 2) * in_chans,
                kernel_size=1,
            ),
            nn.PixelShuffle(patch_size),
        )

    def forward(self, x: torch.Tensor, mask_rgb: torch.Tensor, mask_nir: torch.Tensor):
        t_rgb = self.image_encoder.patch_embed(x[:, :3])
        t_nir = self.image_encoder.patch_embed_nir(x[:, 3:4])

        if mask_rgb.dim() == 3:
            mask_rgb = mask_rgb.unsqueeze(-1)
        if mask_nir.dim() == 3:
            mask_nir = mask_nir.unsqueeze(-1)

        t_rgb = t_rgb * (1.0 - mask_rgb) + self.mask_token_rgb * mask_rgb
        t_nir = t_nir * (1.0 - mask_nir) + self.mask_token_nir * mask_nir
        t = t_rgb + t_nir

        if self.image_encoder.pos_embed is not None:
            pos = F.interpolate(
                self.image_encoder.pos_embed.permute(0, 3, 1, 2),
                size=(t.shape[1], t.shape[2]),
                mode="bicubic",
                align_corners=False,
            ).permute(0, 2, 3, 1)
            t = t + pos

        for blk in self.image_encoder.blocks:
            t = blk(t)

        z = self.image_encoder.neck(t.permute(0, 3, 1, 2))
        x_rec = self.decoder(z)

        mask_rgb_tok = mask_rgb.squeeze(-1) if mask_rgb.dim() == 4 else mask_rgb
        mask_nir_tok = mask_nir.squeeze(-1) if mask_nir.dim() == 4 else mask_nir
        pix_mask_rgb = (
            mask_rgb_tok
            .repeat_interleave(self.patch_size, dim=1)
            .repeat_interleave(self.patch_size, dim=2)
            .unsqueeze(1)
        )
        pix_mask_nir = (
            mask_nir_tok
            .repeat_interleave(self.patch_size, dim=1)
            .repeat_interleave(self.patch_size, dim=2)
            .unsqueeze(1)
        )

        loss_map = F.l1_loss(x, x_rec, reduction="none")
        loss_rgb = (loss_map[:, :3] * pix_mask_rgb).sum() / (pix_mask_rgb.sum() * 3 + 1e-5)
        loss_nir = (loss_map[:, 3:4] * pix_mask_nir).sum() / (pix_mask_nir.sum() + 1e-5)
        loss = loss_rgb + self.lambda_nir * loss_nir

        return loss, x_rec, loss_rgb, loss_nir

    def no_weight_decay(self):
        return {"mask_token_rgb", "mask_token_nir"}
