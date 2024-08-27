import math
from re import T
import numpy as np
import pdb
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, List
from torch.profiler import profile, record_function, ProfilerActivity
from collections import defaultdict

# from gpu_mem_track import MemTracker
from einops import rearrange, repeat

from basicsr.archs.vqgan_arch import Encoder, VectorQuantizer, GumbelQuantizer, Generator, ResBlock
from basicsr.archs.arch_util import flow_warp, resize_flow
from basicsr.archs.pwcnet_arch import FlowGenerator
from basicsr.utils import get_root_logger
from basicsr.utils.registry import ARCH_REGISTRY

from diffusers.models.attention import CrossAttention, FeedForward, AdaLayerNorm

# gpu_tracker = MemTracker()


def calc_mean_std(feat, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.

    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    """Adaptive instance normalization.

    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.

    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)
                       ) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)),
                               device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats,
                             dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerSALayer(nn.Module):
    def __init__(self, embed_dim, nhead=8, dim_mlp=2048, dropout=0.0, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model - MLP
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

        # self attention
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        # ffn
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt


class Fuse_sft_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.encode_enc = ResBlock(2*in_ch, out_ch)

        self.scale = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

        self.shift = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            module.weight.data.zero_()
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, enc_feat, dec_feat, w=1):
        # print(enc_feat.shape, dec_feat.shape)
        enc_feat = self.encode_enc(torch.cat([enc_feat, dec_feat], dim=1))
        scale = self.scale(enc_feat)
        shift = self.shift(enc_feat)
        residual = w * (dec_feat * scale + shift)
        out = dec_feat + residual
        return out


class CrossFrameFusionLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        upcast_attention: bool = False,
    ):
        super().__init__()
        self.use_ada_layer_norm = num_embeds_ada_norm is not None

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Feed-forward
        self.ff = FeedForward(dim, dropout=dropout,
                              activation_fn=activation_fn)

        # Cross Frame Attention
        self.attn = CrossAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )
        nn.init.zeros_(self.attn.to_out[0].weight.data)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.zero_()
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, curr_states, prev_states, residual=True):
        B, C, H, W = curr_states.shape
        curr_states = rearrange(curr_states, "b c h w -> b (h w) c")
        prev_states = rearrange(prev_states, "b c h w -> b (h w) c")

        if residual:
            res = curr_states

        curr_states = self.attn(curr_states, prev_states)
        curr_states = self.norm1(curr_states)

        if residual:
            curr_states = curr_states + res
            res = curr_states

        curr_states = self.ff(curr_states)
        curr_states = self.norm2(curr_states)

        if residual:
            curr_states = curr_states + res

        curr_states = rearrange(curr_states, "b (h w) c -> b c h w", h=H)
        return curr_states


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None

        # SC-Attn
        self.attn1 = SparseCausalAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )
        self.norm1 = AdaLayerNorm(
            dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

        # # Cross-Attn
        # if cross_attention_dim is not None:
        #     self.attn2 = CrossAttention(
        #         query_dim=dim,
        #         cross_attention_dim=cross_attention_dim,
        #         heads=num_attention_heads,
        #         dim_head=attention_head_dim,
        #         dropout=dropout,
        #         bias=attention_bias,
        #         upcast_attention=upcast_attention,
        #     )
        # else:
        #     self.attn2 = None

        # if cross_attention_dim is not None:
        #     self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        # else:
        #     self.norm2 = None

        # Feed-forward
        self.ff = FeedForward(dim, dropout=dropout,
                              activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)

        # Temp-Attn
        self.attn_temp = CrossAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )
        nn.init.zeros_(self.attn_temp.to_out[0].weight.data)
        self.norm_temp = AdaLayerNorm(
            dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool):
        if not is_xformers_available():
            print("Here is how to install it")
            raise ModuleNotFoundError(
                "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                " xformers",
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
                " available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e
            self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            if self.attn2 is not None:
                self.attn2._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            # self.attn_temp._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, attention_mask=None, video_length=None):
        # SparseCausal-Attention
        norm_hidden_states = (
            self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(
                hidden_states)
        )

        if self.only_cross_attention:
            hidden_states = (
                self.attn1(norm_hidden_states, encoder_hidden_states,
                           attention_mask=attention_mask) + hidden_states
            )
        else:
            hidden_states = self.attn1(
                norm_hidden_states, attention_mask=attention_mask, video_length=video_length) + hidden_states

        # if self.attn2 is not None:
        #     # Cross-Attention
        #     norm_hidden_states = (
        #         self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
        #     )
        #     hidden_states = (
        #         self.attn2(
        #             norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
        #         )
        #         + hidden_states
        #     )

        # Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        # Temporal-Attention
        d = hidden_states.shape[1]
        hidden_states = rearrange(
            hidden_states, "(b f) d c -> (b d) f c", f=video_length)
        norm_hidden_states = (
            self.norm_temp(hidden_states, timestep) if self.use_ada_layer_norm else self.norm_temp(
                hidden_states)
        )
        hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
        hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states


class SparseCausalAttention(CrossAttention):
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None):
        batch_size, sequence_length, _ = hidden_states.shape

        if self.group_norm is not None:
            hidden_states = self.group_norm(
                hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        dim = query.shape[-1]
        query = self.reshape_heads_to_batch_dim(query)

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        former_frame_index = torch.arange(video_length) - 1
        former_frame_index[0] = 0

        # d = h*w
        key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
        key = torch.cat([key[:, [0] * video_length],
                        key[:, former_frame_index]], dim=2)
        key = rearrange(key, "b f d c -> (b f) d c")

        value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
        value = torch.cat([value[:, [0] * video_length],
                          value[:, former_frame_index]], dim=2)
        value = rearrange(value, "b f d c -> (b f) d c")

        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(
                    attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(
                    self.heads, dim=0)

        # attention, what we cannot get enough of
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(
                query, key, value, attention_mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(
                    query, key, value, attention_mask)
            else:
                hidden_states = self._sliced_attention(
                    query, key, value, sequence_length, dim, attention_mask)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


class KalmanFilter(nn.Module):
    def __init__(self, emb_dim, num_attention_heads,
                 attention_head_dim, num_uncertainty_layers):
        super().__init__()
        self.uncertainty_estimator = nn.ModuleList(
            [
                BasicTransformerBlock(
                    emb_dim,
                    num_attention_heads,
                    attention_head_dim,
                )
                for d in range(num_uncertainty_layers)
            ]
        )

        self.kalman_gain_calculator = nn.Sequential(
            ResBlock(emb_dim, emb_dim),
            ResBlock(emb_dim, emb_dim),
            ResBlock(emb_dim, emb_dim),
            nn.Conv2d(emb_dim, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def predict(self, z_hat, flow):
        # Predict the next state based on the current state and flow (if available)
        flow = rearrange(flow, "n c h w -> n h w c")
        z_prime = flow_warp(z_hat, flow)
        return z_prime

    def update(self, z_code, z_prime, gain):
        # Update the state and uncertainty based on the measurement and Kalman gain
        z_hat = (1 - gain) * z_code + gain * z_prime
        return z_hat

    def calc_gain(self, z_codes):
        assert z_codes.dim(
        ) == 5, f"Expected z_codes to have ndim=5, but got ndim={z_codes.dim()}."
        video_length = z_codes.shape[1]
        height, width = z_codes.shape[3:5]

        # Assume input shape of uncertainty_estimator to be [(b f) d c]
        z_tmp = rearrange(z_codes, "b f c h w -> (b f) (h w) c")
        h_codes = z_tmp
        for block in self.uncertainty_estimator:
            h_codes = block(h_codes, video_length=video_length)

        h_codes = rearrange(
            h_codes, "(b f) (h w) c -> (b f) c h w", h=height, f=video_length)
        w_codes = self.kalman_gain_calculator(h_codes)

        w_codes = rearrange(
            w_codes, "(b f) c h w -> b f c h w", f=video_length)

        # pdb.set_trace()
        return w_codes


@ARCH_REGISTRY.register()
class KEEP(nn.Module):
    def __init__(self, img_size=512, nf=64, ch_mult=[1, 2, 2, 4, 4, 8], quantizer_type="nearest",
                 res_blocks=2, attn_resolutions=[16], codebook_size=1024, emb_dim=256,
                 beta=0.25, gumbel_straight_through=False, gumbel_kl_weight=1e-8, vqgan_path=None,
                 dim_embd=512, n_head=8, n_layers=9, latent_size=256,
                 connect_list=['32', '64', '128', '256'], fix_modules=['quantize', 'generator'],
                 flow_type='pwc', flownet_path=None, kalman_attn_head_dim=64, num_uncertainty_layers=4,
                 cond=1, cross_fuse_list=[], cross_fuse_nhead=4, cross_fuse_dim=256,
                 cross_fuse_nlayers=4, cross_residual=True,
                 temp_reg_list=[], mask_ratio=0.):
        super().__init__()

        self.cond = cond
        self.connect_list = connect_list
        self.cross_fuse_list = cross_fuse_list
        self.temp_reg_list = temp_reg_list
        self.use_residual = cross_residual
        self.mask_ratio = mask_ratio
        self.latent_size = latent_size
        logger = get_root_logger()

        # alignment
        # assert flownet_path != None, "Missing path to load pre-trained flow net."
        if flow_type == 'pwc':
            from basicsr.archs.pwcnet_arch import FlowGenerator
            self.flownet = FlowGenerator(path=flownet_path)
        elif flow_type == 'gmflow':
            from basicsr.archs.gmflow_arch import FlowGenerator
            self.flownet = FlowGenerator(path=flownet_path)

        # Kalman Filter
        self.kalman_filter = KalmanFilter(
            emb_dim=emb_dim,
            num_attention_heads=n_head,
            attention_head_dim=kalman_attn_head_dim,
            num_uncertainty_layers=num_uncertainty_layers,
        )

        self.hq_encoder = Encoder(
            in_channels=3,
            nf=nf,
            emb_dim=emb_dim,
            ch_mult=ch_mult,
            num_res_blocks=res_blocks,
            resolution=img_size,
            attn_resolutions=attn_resolutions
        )

        # VQGAN
        self.encoder = Encoder(
            in_channels=3,
            nf=nf,
            emb_dim=emb_dim,
            ch_mult=ch_mult,
            num_res_blocks=res_blocks,
            resolution=img_size,
            attn_resolutions=attn_resolutions
        )
        if quantizer_type == "nearest":
            self.quantize = VectorQuantizer(codebook_size, emb_dim, beta)
        elif quantizer_type == "gumbel":
            self.quantize = GumbelQuantizer(
                codebook_size, emb_dim, emb_dim, gumbel_straight_through, gumbel_kl_weight
            )
        self.generator = Generator(
            nf=nf,
            emb_dim=emb_dim,
            ch_mult=ch_mult,
            res_blocks=res_blocks,
            img_size=img_size,
            attn_resolutions=attn_resolutions
        )

        if vqgan_path is not None:
            ckpt = torch.load(vqgan_path, map_location='cpu')
            if 'params_ema' in ckpt:
                self.load_state_dict(torch.load(
                    vqgan_path, map_location='cpu')['params_ema'], strict=False)
                logger.info(f'vqgan is loaded from: {vqgan_path} [params_ema]')
            elif 'params' in ckpt:
                self.load_state_dict(torch.load(
                    vqgan_path, map_location='cpu')['params'], strict=False)
                logger.info(f'vqgan is loaded from: {vqgan_path} [params]')
            else:
                raise ValueError(f'Wrong params!')

        self.position_emb = nn.Parameter(torch.zeros(latent_size, dim_embd))
        self.feat_emb = nn.Linear(emb_dim, dim_embd)

        # transformer
        self.ft_layers = nn.Sequential(*[TransformerSALayer(embed_dim=dim_embd, nhead=n_head,
                                                            dim_mlp=dim_embd*2, dropout=0.0) for _ in range(n_layers)])

        # logits_predict head
        self.idx_pred_layer = nn.Sequential(
            nn.LayerNorm(dim_embd),
            nn.Linear(dim_embd, codebook_size, bias=False))

        self.channels = {
            '16': 512,
            '32': 256,
            '64': 256,
            '128': 128,
            '256': 128,
            '512': 64,
        }

        # after second residual block for > 16, before attn layer for ==16
        self.fuse_encoder_block = {'512': 2, '256': 5,
                                   '128': 8, '64': 11, '32': 14, '16': 18}
        # after first residual block for > 16, before attn layer for ==16
        self.fuse_generator_block = {
            '16': 6, '32': 9, '64': 12, '128': 15, '256': 18, '512': 21}

        # cross frame attention fusion
        self.cross_fuse = nn.ModuleDict()
        for f_size in self.cross_fuse_list:
            in_ch = self.channels[f_size]
            self.cross_fuse[f_size] = CrossFrameFusionLayer(dim=in_ch,
                                                            num_attention_heads=cross_fuse_nhead,
                                                            attention_head_dim=cross_fuse_dim)

        # fuse_convs_dict
        self.fuse_convs_dict = nn.ModuleDict()
        for f_size in self.connect_list:
            in_ch = self.channels[f_size]
            self.fuse_convs_dict[f_size] = Fuse_sft_block(in_ch, in_ch)

        if fix_modules is not None:
            for module in fix_modules:
                for param in getattr(self, module).parameters():
                    param.requires_grad = False

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_flow(self, x):
        b, t, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        # Forward flow
        with torch.no_grad():
            flows = self.flownet(x_2, x_1).view(b, t - 1, 2, h, w)

        return flows.detach()

    def mask_by_ratio(self, x, mask_ratio=0.):
        if mask_ratio == 0:
            return x

        # B F C H W
        b, t, c, h, w = x.size()
        d = h * w
        x = rearrange(x, "b f c h w -> b f (h w) c")

        len_keep = int(d * (1 - mask_ratio))
        sample = torch.rand((b, t, d, 1), device=x.device).topk(
            len_keep, dim=2).indices
        mask = torch.zeros((b, t, d, 1), dtype=torch.bool, device=x.device)
        mask.scatter_(dim=2, index=sample, value=True)

        x = mask * x
        x = rearrange(x, "b f (h w) c -> b f c h w", h=h)

        return x

    def forward(self, x, detach_16=True, early_feat=True, need_upscale=True):
        """Forward function for KEEP.

        Args:
            lqs (tensor): Input low quality (LQ) sequence of
                shape (b, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (b, t, c, 4h, 4w).
        """
        video_length = x.shape[1]

        if need_upscale:
            x = rearrange(x, "b f c h w -> (b f) c h w")
            x = F.interpolate(x, scale_factor=4, mode='bilinear')
            x = rearrange(x, "(b f) c h w -> b f c h w", f=video_length)

        b, t, c, h, w = x.size()
        flows = self.get_flow(x)  # (B, t-1, 2, H , W)

        # ################### Encoder #####################
        # BTCHW -> (BT)CHW
        x = x.reshape(-1, c, h, w)
        enc_feat_dict = {}
        out_list = [self.fuse_encoder_block[f_size]
                    for f_size in self.connect_list]
        for i, block in enumerate(self.encoder.blocks):
            x = block(x)
            if i in out_list:
                enc_feat_dict[str(x.shape[-1])] = x.detach()

        lq_feat = x

        # gpu_tracker.track('After encoder')
        # ################### Kalman Filter ###############
        z_codes = rearrange(x, "(b f) c h w -> b f c h w", f=t)
        if self.training:
            z_codes = self.mask_by_ratio(z_codes, self.mask_ratio)
        gains = self.kalman_filter.calc_gain(z_codes)

        outs = []
        logits = []
        cross_prev_feat = {}
        gen_feat_dict = defaultdict(list)

        fuse_list = [self.fuse_generator_block[f_size]
                     for f_size in self.connect_list]

        cross_fuse_list = [self.fuse_generator_block[f_size]
                           for f_size in self.cross_fuse_list]

        temp_reg_list = [self.fuse_generator_block[f_size]
                         for f_size in self.temp_reg_list]

        for i in range(video_length):
            # print(f'Frame {i} ...')
            if i == 0:
                z_hat = z_codes[:, i, ...]
                # gpu_tracker.track()
            else:
                # gpu_tracker.track(f'Before {i}-frame Kalman Filter')
                z_prime = self.hq_encoder(
                    self.kalman_filter.predict(prev_out.detach(), flows[:, i-1, ...]))
                z_hat = self.kalman_filter.update(
                    z_codes[:, i, ...], z_prime, gains[:, i, ...])
                # z_hat = z_codes[:, i, ...]
                # gpu_tracker.track(f'After {i}-frame Kalman Filter')
                # del z_prime
                # torch.cuda.empty_cache()

            # ################# Transformer ###################
            pos_emb = self.position_emb.unsqueeze(1).repeat(1, b, 1)
            # BCHW -> BC(HW) -> (HW)BC
            # pdb.set_trace()
            query_emb = self.feat_emb(z_hat.flatten(2).permute(2, 0, 1))
            # print(pos_emb.shape, query_emb.shape)
            for layer in self.ft_layers:
                query_emb = layer(query_emb, query_pos=pos_emb)

            # output logits
            logit = self.idx_pred_layer(query_emb).permute(
                1, 0, 2)  # (hw)bn -> b(hw)n
            logits.append(logit)

            # ################# Quantization ###################
            code_h = int(np.sqrt(self.latent_size))
            soft_one_hot = F.softmax(logit, dim=2)
            _, top_idx = torch.topk(soft_one_hot, 1, dim=2)
            quant_feat = self.quantize.get_codebook_feat(
                top_idx, shape=[b, code_h, code_h, 256])

            if detach_16:
                # for training stage III
                quant_feat = quant_feat.detach()
            else:
                # preserve gradients for stage II
                quant_feat = z_hat + (quant_feat - z_hat).detach()

            # ################## Generator ####################
            x = quant_feat

            for j, block in enumerate(self.generator.blocks):
                x = block(x)

                if j in fuse_list:  # fuse after i-th block
                    f_size = str(x.shape[-1])
                    x = self.fuse_convs_dict[f_size](
                        enc_feat_dict[f_size][i: i+1, ...], x, self.cond)

                if j in cross_fuse_list:
                    f_size = str(x.shape[-1])

                    if i == 0:
                        cross_prev_feat[f_size] = x
                        # print(f_size)
                    else:
                        # pdb.set_trace()
                        prev_fea = cross_prev_feat[f_size]
                        x = self.cross_fuse[f_size](
                            x, prev_fea, residual=self.use_residual)
                        cross_prev_feat[f_size] = x

                if j in temp_reg_list:
                    f_size = str(x.shape[-1])
                    gen_feat_dict[f_size].append(x)

            prev_out = x  # B C H W
            outs.append(prev_out)

        for f_size, feat in gen_feat_dict.items():
            gen_feat_dict[f_size] = torch.stack(feat, dim=1)  # bfchw

        logits = torch.stack(logits, dim=1)  # b(hw)n -> bf(hw)n
        logits = rearrange(logits, "b f l n -> (b f) l n")
        outs = torch.stack(outs, dim=1)  # bfchw
        if self.training:
            if early_feat:
                return outs, logits, lq_feat, gen_feat_dict
            else:
                return outs, gen_feat_dict
        else:
            return outs


def count_parameters(model):
    # Initialize counters
    total_params = 0
    sub_module_params = {}

    # Loop through all the modules in the model
    for name, module in model.named_children():
        # if len(list(module.children())) == 0:  # Check if it's a leaf module
        params = sum(p.numel() for p in module.parameters())
        total_params += params
        sub_module_params[name] = params

    return total_params, sub_module_params


if __name__ == '__main__':
    import time
    batch_size = 1
    video_length = 4
    height = 128
    width = 128

    model = KEEP(
        img_size=512,
        emb_dim=256,
        ch_mult=[1, 2, 2, 4, 4, 8],
        dim_embd=512,
        n_head=8,
        n_layers=4,
        codebook_size=1024,
        connect_list=[],
        fix_modules=['generator', 'quantize', 'flownet', 'fuse_convs_dict', 'hq_encoder',
                     'encoder', 'feat_emb', 'ft_layers', 'idx_pred_layer'],
        flow_type='gmflow',
        flownet_path="../../weights/GMFlow/gmflow_sintel-0c07dcb3.pth",
        kalman_attn_head_dim=32,
        num_uncertainty_layers=3,
        cond=0,
        cross_fuse_list=['32'],
        cross_fuse_nhead=4,
        cross_fuse_dim=256,
        temp_reg_list=['64'],
    ).cuda()

    total_params = sum(map(lambda x: x.numel(), model.parameters()))
    print(f"Total parameters in the model: {total_params / 1e6:.2f} M")

    # print(f"Total parameters in the model: {total_params / 1e6:.2f} G")
    # for name, params in sub_module_params.items():
    #     print(f"Parameters in {name}: {params / 1e6:.2f} G")
    #
    # ckpt_path = '../../weights/CodeFormer/codeformer_doubleEnc_vqgan800k.pth'
    # checkpoint = torch.load(ckpt_path)['params_ema']
    # model.load_state_dict(checkpoint, strict=False)

    dummy_input = torch.randn((1, 20, 3, 128, 128)).cuda()

    start_time = time.time()

    with torch.no_grad():
        for _ in range(100):
            out = model(dummy_input)
    elapsed_time = time.time() - start_time

    print(f"Forward pass time: {elapsed_time / 100 / 20 * 1000:.2f} ms")
    print(out.shape)
