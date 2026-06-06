# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder


class Sam(nn.Module):
    """SAM backbone repurposed as a prompt-free semantic segmenter.

    Forward pipeline (matches the training/validation logic that previously
    lived in ``function.py``):

        ``image_encoder`` -> ``prompt_encoder(no prompts, frozen)``
        -> ``mask_decoder`` -> bilinear upsample to ``args.out_size``.

    The prompt encoder is run inside ``torch.no_grad()`` so its parameters
    are never updated by autograd, mimicking the original fine-tuning script.
    """

    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        args,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
    ) -> None:
        super().__init__()
        self.args = args
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Prompt-free segmentation forward.

        Args:
            x: Input batch with shape ``[B, C, H, W]``. ``C`` may be 3 (RGB) or
               4 (RGB + NIR) -- the underlying ``ImageEncoderViT`` handles both.

        Returns:
            ``logits`` with shape ``[B, num_classes, out_size, out_size]``.
            ``out_size`` defaults to ``self.image_encoder.img_size`` if not present
            on ``args``.
        """
        image_embeddings = self.image_encoder(x)

        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )

        multimask_output = int(getattr(self.args, "num_classes", 1)) > 1
        low_res_logits, _ = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        out_size = int(getattr(self.args, "out_size", self.image_encoder.img_size))
        logits = F.interpolate(
            low_res_logits,
            size=(out_size, out_size),
            mode="bilinear",
            align_corners=False,
        )
        return logits
