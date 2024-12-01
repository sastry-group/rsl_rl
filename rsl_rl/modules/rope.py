from __future__ import annotations
from math import pi, log

import torch
from torch.amp import autocast
from torch.nn import Module, ModuleList
from torch import nn, einsum, broadcast_tensors, Tensor

from einops import rearrange, repeat

from typing import Literal

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# broadcat, as tortoise-tts was using it

def broadcat(tensors, dim = -1):
    broadcasted_tensors = broadcast_tensors(*tensors)
    return torch.cat(broadcasted_tensors, dim = dim)

def slice_at_dim(t, dim_slice: slice, *, dim):
    dim += (t.ndim if dim < 0 else 0)
    colons = [slice(None)] * t.ndim
    colons[dim] = dim_slice
    return t[tuple(colons)]

# rotary embedding helper functions

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

@autocast('cuda', enabled = False)
def apply_rotary_emb(
    freqs,
    t,
    start_index = 0,
    scale = 1.,
    seq_dim = -2,
    freqs_seq_dim = None
):
    dtype = t.dtype

    if not exists(freqs_seq_dim):
        if freqs.ndim == 2 or t.ndim == 3:
            freqs_seq_dim = 0

    if t.ndim == 3 or exists(freqs_seq_dim):
        seq_len = t.shape[seq_dim]
        freqs = slice_at_dim(freqs, slice(-seq_len, None), dim = freqs_seq_dim)

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'

    # Split t into three parts: left, middle (to be transformed), and right
    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]

    # Apply rotary embeddings without modifying t in place
    t_transformed = (t_middle * freqs.cos() * scale) + (rotate_half(t_middle) * freqs.sin() * scale)

    out = torch.cat((t_left, t_transformed, t_right), dim=-1)

    return out.type(dtype)

# learned rotation helpers

def apply_learned_rotations(rotations, t, start_index = 0, freq_ranges = None):
    if exists(freq_ranges):
        rotations = einsum('..., f -> ... f', rotations, freq_ranges)
        rotations = rearrange(rotations, '... r f -> ... (r f)')

    rotations = repeat(rotations, '... n -> ... (n r)', r = 2)
    return apply_rotary_emb(rotations, t, start_index = start_index)

# classes

class RotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        custom_freqs: Tensor | None = None,
        freqs_for:  Literal['lang', 'pixel', 'constant'] = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
        learned_freq = False,
        use_xpos = False,
        xpos_scale_base = 512,
        interpolate_factor = 1.,
        theta_rescale_factor = 1.,
        seq_before_head_dim = False,
        cache_if_possible = True,
        cache_max_seq_len = 8192
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/

        theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.freqs_for = freqs_for

        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()

        self.cache_if_possible = cache_if_possible
        self.cache_max_seq_len = cache_max_seq_len

        self.register_buffer('cached_freqs', torch.zeros(cache_max_seq_len, dim), persistent = False)
        self.register_buffer('cached_freqs_seq_len', torch.tensor(0), persistent = False)

        self.freqs = nn.Parameter(freqs, requires_grad = learned_freq)

        self.learned_freq = learned_freq

        # dummy for device

        self.register_buffer('dummy', torch.tensor(0), persistent = False)

        # default sequence dimension

        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        # interpolation factors

        assert interpolate_factor >= 1.
        self.interpolate_factor = interpolate_factor

        # xpos

        self.use_xpos = use_xpos

        if not use_xpos:
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base

        self.register_buffer('scale', scale, persistent = False)
        self.register_buffer('cached_scales', torch.zeros(cache_max_seq_len, dim), persistent = False)
        self.register_buffer('cached_scales_seq_len', torch.tensor(0), persistent = False)

        # add apply_rotary_emb as static method

        self.apply_rotary_emb = staticmethod(apply_rotary_emb)

    @property
    def device(self):
        return self.dummy.device

    def get_seq_pos(self, seq_len, device, dtype, offset = 0):
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        if torch.is_tensor(offset):
            positions = positions.unsqueeze(0)  # shape (1, seq_len)
            offset = offset.unsqueeze(1)  # shape (batch_size, 1)
            seq = (positions + offset) / self.interpolate_factor  # shape (batch_size, seq_len)
        else:
            seq = (positions + offset) / self.interpolate_factor  # shape (seq_len,)
        return seq  # shape is (batch_size, seq_len) if offset is tensor, else (seq_len,)

    def rotate_queries_or_keys(self, t, seq_dim = None, offset = 0, scale = None):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert not self.use_xpos or exists(scale), 'you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings'

        device, dtype = t.device, t.dtype

        seq_len = t.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, device=device, dtype=dtype, offset=offset)

        freqs = self.forward(seq, seq_len=seq_len)

        # Expand freqs to match t's dimensions
        # t shape: (batch_size, num_heads, seq_len, head_dim)
        # freqs shape: (batch_size, seq_len, head_dim) or (seq_len, head_dim)

        if freqs.ndim == 2:
            # freqs shape: (seq_len, head_dim)
            freqs = freqs.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, seq_len, head_dim)
        else:
            # freqs shape: (batch_size, seq_len, head_dim)
            freqs = freqs.unsqueeze(1)  # shape: (batch_size, 1, seq_len, head_dim)

        # Now freqs can broadcast over batch_size and num_heads

        if exists(scale):
            if torch.is_tensor(scale):
                if scale.ndim == 2:
                    # scale shape: (batch_size, seq_len)
                    scale = scale.unsqueeze(1).unsqueeze(-1)  # shape: (batch_size, 1, seq_len, 1)
                elif scale.ndim == 1:
                    # scale shape: (seq_len,)
                    scale = scale.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # shape: (1, 1, seq_len, 1)
            else:
                # scale is scalar
                pass
        else:
            scale = 1.

        return apply_rotary_emb(freqs, t, scale=scale, seq_dim=seq_dim)

    def rotate_queries_with_cached_keys(self, q, k, seq_dim = None, offset = 0):
        dtype, device, seq_dim = q.dtype, q.device, default(seq_dim, self.default_seq_dim)

        q_len, k_len = q.shape[seq_dim], k.shape[seq_dim]
        assert q_len <= k_len

        q_scale = k_scale = 1.

        if self.use_xpos:
            seq = self.get_seq_pos(k_len, dtype=dtype, device=device, offset=0)
            q_scale = self.get_scale(seq[:, -q_len:]).type(dtype)
            k_scale = self.get_scale(seq).type(dtype)

        rotated_q = self.rotate_queries_or_keys(q, seq_dim=seq_dim, scale=q_scale, offset=k_len - q_len + offset)
        rotated_k = self.rotate_queries_or_keys(k, seq_dim=seq_dim, scale=k_scale ** -1, offset=offset)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def rotate_queries_and_keys(self, q, k, seq_dim = None):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert self.use_xpos
        device, dtype = q.device, q.dtype

        seq_len = q.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, dtype=dtype, device=device)

        freqs = self.forward(seq, seq_len=seq_len)
        scale = self.get_scale(seq, seq_len=seq_len).to(dtype)

        # Expand freqs and scale to match q and k
        if freqs.ndim == 2:
            freqs = freqs.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, seq_len, head_dim)
            scale = scale.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # shape: (1, 1, seq_len, 1)
        else:
            freqs = freqs.unsqueeze(1)  # shape: (batch_size, 1, seq_len, head_dim)
            scale = scale.unsqueeze(1).unsqueeze(-1)  # shape: (batch_size, 1, seq_len, 1)

        rotated_q = apply_rotary_emb(freqs, q, scale=scale, seq_dim=seq_dim)
        rotated_k = apply_rotary_emb(freqs, k, scale=scale ** -1, seq_dim=seq_dim)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def get_scale(
        self,
        t: Tensor,
        seq_len: int | None = None,
        offset = 0
    ):
        assert self.use_xpos

        # Disable caching if offset is a tensor
        if torch.is_tensor(offset):
            should_cache = False
        else:
            should_cache = (
                self.cache_if_possible and
                exists(seq_len) and
                (offset + seq_len) <= self.cache_max_seq_len
            )

        if (
            should_cache and
            exists(self.cached_scales) and
            (seq_len + offset) <= self.cached_scales_seq_len.item()
        ):
            return self.cached_scales[offset:(offset + seq_len)]

        scale = 1.
        if self.use_xpos:
            power = (t - t.shape[-1] // 2) / self.scale_base
            scale = self.scale ** rearrange(power, '... -> ... 1')
            scale = repeat(scale, '... d -> ... (d r)', r = 2)

        if should_cache and offset == 0:
            self.cached_scales[:seq_len] = scale.detach()
            self.cached_scales_seq_len.copy_(seq_len)

        return scale

    @autocast('cuda', enabled = False)
    def forward(
        self,
        t: Tensor,
        seq_len = None
    ):
        # Disable caching if t has batch dimension (i.e., offset is a tensor)
        if t.ndim > 1:
            should_cache = False
        else:
            should_cache = (
                self.cache_if_possible and
                not self.learned_freq and
                exists(seq_len) and
                self.freqs_for != 'pixel' and
                seq_len <= self.cache_max_seq_len
            )

        if (
            should_cache and
            exists(self.cached_freqs) and
            seq_len <= self.cached_freqs_seq_len.item()
        ):
            return self.cached_freqs[:seq_len].detach()

        freqs = self.freqs

        freqs = einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)

        if should_cache:
            self.cached_freqs[:seq_len] = freqs.detach()
            self.cached_freqs_seq_len.copy_(seq_len)

        return freqs

    def get_axial_freqs(self, *dims):
        Colon = slice(None)
        all_freqs = []

        for ind, dim in enumerate(dims):
            if self.freqs_for == 'pixel':
                pos = torch.linspace(-1, 1, steps = dim, device = self.device)
            else:
                pos = torch.arange(dim, device = self.device)

            seq_len = dim
            seq_pos = self.get_seq_pos(seq_len, device=self.device, dtype=pos.dtype, offset=0)
            freqs = self.forward(seq_pos, seq_len=seq_len)

            all_axis = [None] * len(dims)
            all_axis[ind] = Colon

            new_axis_slice = (Ellipsis, *all_axis, Colon)
            all_freqs.append(freqs[new_axis_slice])

        all_freqs = broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim = -1)
