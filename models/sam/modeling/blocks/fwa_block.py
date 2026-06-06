from typing import Any, Dict, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


class FWABlock(nn.Module):
    """
    Transformer block with FreqWeaverAdapter after self-attention (spatial branch)
    and an MLP Adapter on norm2 features in the FFN branch (standard adapter tuning).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        scale: float = 0.5,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
        mid_dim: Optional[int] = None,
        freq_weaver_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        adapter_dim = mid_dim if mid_dim is not None else dim

        self.MLP_Adapter = Adapter(adapter_dim, skip_connect=False)
        fw_kw = dict(freq_weaver_kwargs or {})
        self.Space_Adapter = FreqWeaverAdapter(adapter_dim, **fw_kw)
        self.scale = scale
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.norm1(x)
        x = self.attn(x)
        x = self.Space_Adapter(x)

        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        xn = self.norm2(x)
        x = x + self.mlp(xn) + self.scale * self.MLP_Adapter(xn)
        return x


class FreqWeaverAdapter(nn.Module):
    """Freq split (high/low) plus spatial MLP branch; router fusion injected as residual [B, H, W, C]."""

    def __init__(
        self,
        dim: int,
        scale: float = 1.0,
        cutoff_freq_ratio: float = 0.15,
        hf_kernel_size: int = 5,
        lf_kernel_size: int = 11,
        conv_groups: int = -1,
        spatial_mlp_bottleneck_ratio: float = 0.25,
        use_router: bool = True,
        router_bottleneck_ratio: float = 0.125,
        activation_fn_class: type = nn.GELU,
    ) -> None:
        super().__init__()
        self.spatial_mlp_bottleneck_ratio = spatial_mlp_bottleneck_ratio

        if dim <= 0:
            raise ValueError(f"Input dimension 'dim' must be positive, got {dim}")

        self.dim = dim
        self.scale = scale
        self.cutoff_freq_ratio = cutoff_freq_ratio
        self.use_router = use_router
        self.register_buffer("_mask", None, persistent=False)

        if conv_groups == -1:
            self.conv_groups = dim
        elif conv_groups == 1:
            self.conv_groups = 1
        elif conv_groups > 0 and dim % conv_groups == 0:
            self.conv_groups = conv_groups
        else:
            raise ValueError(
                f"conv_groups must be 1, -1 (for depthwise), or a divisor of dim. "
                f"Got {conv_groups} for dim {dim}"
            )

        self.spatial_adapter = Adapter(
            dim, mlp_ratio=spatial_mlp_bottleneck_ratio, skip_connect=False
        )

        self.hf_adapter = nn.Conv2d(
            dim,
            dim,
            kernel_size=hf_kernel_size,
            padding=hf_kernel_size // 2,
            groups=self.conv_groups,
        )
        self.lf_adapter = nn.Conv2d(
            dim,
            dim,
            kernel_size=lf_kernel_size,
            padding=lf_kernel_size // 2,
            groups=self.conv_groups,
        )

        self.act = activation_fn_class()

        for conv_layer in (self.hf_adapter, self.lf_adapter):
            nn.init.zeros_(conv_layer.weight)
            if conv_layer.bias is not None:
                nn.init.zeros_(conv_layer.bias)

        if self.use_router:
            router_hidden_dim = int(dim * router_bottleneck_ratio)
            if router_hidden_dim <= 0:
                router_hidden_dim = max(1, dim // 8) if dim > 8 else 1
            self.router = nn.Sequential(
                nn.Linear(dim, router_hidden_dim),
                activation_fn_class(),
                nn.Linear(router_hidden_dim, 3),
            )
            if isinstance(self.router[-1], nn.Linear):
                nn.init.zeros_(self.router[-1].weight)
                if self.router[-1].bias is not None:
                    nn.init.zeros_(self.router[-1].bias)
        else:
            self.fusion_weights_param = nn.Parameter(torch.zeros(3))

    def _build_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        freq_y = torch.fft.fftfreq(H, device=device)
        freq_x = torch.fft.fftfreq(W, device=device)
        gy, gx = torch.meshgrid(freq_y, freq_x, indexing="ij")
        radial_freq_sq = gy**2 + gx**2
        mask = radial_freq_sq < (self.cutoff_freq_ratio**2)
        return mask.unsqueeze(0).unsqueeze(0).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not (x.ndim == 4 and x.shape[-1] == self.dim):
            raise ValueError(
                f"Input tensor x must be of shape [B, H, W, C] with C={self.dim}, got {x.shape}"
            )

        x_res = x
        B, H, W, C = x.shape

        delta_s_bhwc = self.spatial_adapter(x)
        x_bchw = x.permute(0, 3, 1, 2).contiguous()

        Fq = torch.fft.fft2(x_bchw, norm="ortho")
        if self._mask is None or self._mask.shape[-2:] != (H, W) or self._mask.device != x.device:
            self._mask = self._build_mask(H, W, x.device)

        M_low_pass = self._mask
        F_lf = Fq * M_low_pass
        F_hf = Fq * (1 - M_low_pass)

        f_lf_bchw = torch.fft.ifft2(F_lf, norm="ortho").real
        f_hf_bchw = torch.fft.ifft2(F_hf, norm="ortho").real

        delta_hf_bchw = self.act(self.hf_adapter(f_hf_bchw))
        delta_lf_bchw = self.act(self.lf_adapter(f_lf_bchw))
        delta_hf_bhwc = delta_hf_bchw.permute(0, 2, 3, 1)
        delta_lf_bhwc = delta_lf_bchw.permute(0, 2, 3, 1)

        if self.use_router:
            pooled_features = x.mean(dim=(1, 2))
            route_logits = self.router(pooled_features)
            route_weights = torch.softmax(route_logits, dim=-1)
            w_s, w_l, w_h = route_weights.unbind(dim=-1)
            fused_delta_bhwc = (
                w_s.view(B, 1, 1, 1) * delta_s_bhwc
                + w_l.view(B, 1, 1, 1) * delta_lf_bhwc
                + w_h.view(B, 1, 1, 1) * delta_hf_bhwc
            )
        else:
            fusion_softmax_weights = torch.softmax(self.fusion_weights_param, dim=-1)
            fused_delta_bhwc = (
                fusion_softmax_weights[0] * delta_s_bhwc
                + fusion_softmax_weights[1] * delta_lf_bhwc
                + fusion_softmax_weights[2] * delta_hf_bhwc
            )

        return x_res + self.scale * fused_delta_bhwc


class Adapter(nn.Module):
    """Bottleneck MLP adapter (same behavior as models.common.adapter)."""

    def __init__(
        self,
        D_features: int,
        mlp_ratio: float = 0.25,
        act_layer: Type[nn.Module] = nn.GELU,
        skip_connect: bool = True,
    ) -> None:
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            return x + xs
        return xs


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, (
                "Input size must be provided if using relative positional encoding."
            )
            self.rel_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_h, self.rel_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)
        return x


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)
    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)
    return attn
