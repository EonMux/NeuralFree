"""Generates the torchao checkpoint fixtures under this directory.

Each fixture is a tiny checkpoint holding one quantized ``model.layers.0.q_proj.weight``
plus a ``bias`` plain tensor, in the form torchao 0.18 writes: a safetensors file whose
``__metadata__`` carries the flatten_tensor_state_dict descriptors, or a PyTorch pickle
whose state dict holds the tensor subclass (``_rebuild_from_type_v2`` over
``_rebuild_wrapper_subclass``, with ``_rebuild_tensor_v3`` inner views for FP8).

``torchao-reference.json`` records, per fixture:

* ``canonical``: every tensor the loader exposes after decoding the subclass (name,
  dtype, shape and SHA-256 of its little-endian stored bytes), and
* ``dequantized``: SHA-256 of the f32 little-endian logical weight torchao dequantizes,
  in row-major order.

Generating the fixtures needs torch 2.12 or newer and torchao 0.18
(``PYTHONPATH=/path/to/torchao``) and a CUDA device only for the tile-packed int4
packer. The mslk preshuffle and the XPU plain-int32 packer cannot run here (mslk
targets SM90 and torch has no XPU backend), so this script reproduces their steps
directly from the reference sources: ``pack_int4`` and ``int4_row_quantize[_zp]``
from ``mslk/quantize/shuffle.py``, ``unified_encode_int4b`` and ``pack_scale_fp8``
from CUTLASS ``tools/util/mixed_dtype_utils.hpp``, and the
``compute_memory_reordering_atom`` permutation, which was checked against a build of
the pinned CUTLASS. Reading the fixtures needs neither torch nor a GPU.

Usage: python3 torchao_fixtures.py [--out DIR] [--only NAME,NAME,...]
"""
import argparse
import hashlib
import json
import os
import struct
import sys

import torch

from torchao.quantization.quantize_.workflows.int4.int4_tensor import Int4Tensor
from torchao.quantization.quantize_.workflows.int4.int4_tile_packed_to_4d_tensor import (
    Int4TilePackedTo4dTensor,
)
from torchao.quantization.quantize_.workflows.int4.int4_plain_int32_tensor import (
    Int4PlainInt32Tensor,
)
from torchao.quantization.quantize_.workflows.int4.int4_preshuffled_tensor import (
    Int4PreshuffledTensor,
)
from torchao.quantization.quantize_.workflows.int8.int8_tensor import Int8Tensor
from torchao.quantization.quantize_.workflows.intx.intx_unpacked_to_int8_tensor import (
    IntxUnpackedToInt8Tensor,
)
from torchao.quantization.quantize_.workflows.float8.float8_tensor import Float8Tensor
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
from torchao.prototype.safetensors.safetensors_support import flatten_tensor_state_dict
from torchao.quantization.utils import pack_tinygemm_scales_and_zeros

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")
M = "model.layers.0.q_proj"
W = M + ".weight"
BIAS = M + ".bias"
SAFETENSORS_DTYPES = {
    torch.uint8: "U8",
    torch.int8: "I8",
    torch.int32: "I32",
    torch.float32: "F32",
    torch.bfloat16: "BF16",
    torch.float8_e4m3fn: "F8_E4M3",
    torch.float8_e8m0fnu: "F8_E8M0",
}


def dtype_name(t):
    for k, v in SAFETENSORS_DTYPES.items():
        if t.dtype == k:
            return v
    raise ValueError("no safetensors dtype for " + str(t.dtype))


def raw(t):
    return t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def digest(t):
    return hashlib.sha256(raw(t)).hexdigest()


def f32_digest(values):
    # Normalize signed zero: ggml's block decoders produce +0 where torchao keeps the sign of a zero product.
    b = b"".join(struct.pack("<f", 0.0 if v == 0 else float(v)) for v in values.reshape(-1).tolist())
    return hashlib.sha256(b).hexdigest()


def npy(t):
    return [int(x) for x in t.shape]


# --------------------------------------------------------------------------------------------- safetensors writer

def write_safetensors(path, tensors_data, metadata):
    header = {"__metadata__": {k: str(v) for k, v in metadata.items()}}
    offset = 0
    chunks = []
    for name, t in tensors_data.items():
        data = raw(t)
        header[name] = {"dtype": dtype_name(t), "shape": npy(t), "data_offsets": [offset, offset + len(data)]}
        chunks.append(data)
        offset += len(data)
    text = json.dumps(header, separators=(",", ":"))
    pad = (8 - (len(text) % 8)) % 8
    text += " " * pad
    encoded = text.encode("utf-8")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
        for c in chunks:
            f.write(c)


# --------------------------------------------------------------------------------------------- the tensors

def int4_tensor():
    n, k, g = 8, 16, 8
    codes = torch.tensor([[(i * 5 + j * 3 + 1) & 0xF for j in range(k)] for i in range(n)], dtype=torch.uint8)
    signed = codes.to(torch.int8)
    signed[signed > 7] -= 16
    packed = (signed[:, ::2] & 0xF) | ((signed[:, 1::2] & 0xF) << 4)
    qdata = packed.to(torch.int8)
    rows = k // g
    scale = torch.tensor([[0.25, 0.5, 0.75, 0.25, 0.5, 0.75, 0.25, 0.5],
                          [0.5, 0.25, 0.5, 0.25, 0.75, 0.25, 0.5, 0.25]], dtype=torch.float32)
    zero = torch.tensor([[0.0, 0.25, 0.5, 0.0, 0.25, 0.5, 0.0, 0.25],
                         [0.5, 0.0, 0.25, 0.5, 0.0, 0.25, 0.5, 0.0]], dtype=torch.float32)
    assert scale.shape == (rows, n)
    t = Int4Tensor(qdata, scale, zero, [1, g], torch.Size([n, k]))
    return t, signed.to(torch.float32), scale, zero


def int4_tile_tensor():
    n, k, g = 16, 128, 32
    # torchao pads K to a multiple of 1024 and N to a multiple of 8 before packing.
    k_pad = ((k + 1023) // 1024) * 1024
    n_pad = ((n + 7) // 8) * 8
    groups = (k + g - 1) // g
    stored_groups = k_pad // g
    codes = torch.zeros(n_pad, k_pad, dtype=torch.uint8)
    codes[:n, :k] = torch.tensor([[(i * 3 + j * 7 + 2) & 0xF for j in range(k)] for i in range(n)],
                                 dtype=torch.uint8)
    inp = ((codes[:, ::2] << 4) | codes[:, 1::2]).to(torch.uint8).cuda()
    qdata = torch.ops.aten._convert_weight_to_int4pack(inp.contiguous(), 8).cpu()
    scale = torch.full((stored_groups, n_pad), 0.5, dtype=torch.bfloat16)
    zero = torch.full((stored_groups, n_pad), 0.25, dtype=torch.bfloat16)
    for c in range(groups):
        for i in range(n):
            scale[c, i] = 0.25 if (c + i) % 3 == 0 else 0.5
            zero[c, i] = 0.0 if (c + i) % 2 == 0 else 0.5
    scale_and_zero = pack_tinygemm_scales_and_zeros(scale.t().contiguous(), zero.t().contiguous(),
                                                    torch.bfloat16)
    t = Int4TilePackedTo4dTensor(qdata, scale_and_zero, [1, g], torch.Size([n, k]))
    # tinygemm codes are unsigned 0..15 with quantized value (q - 8) * scale + zero.
    w = torch.empty(n, k, dtype=torch.float32)
    for i in range(n):
        for j in range(k):
            w[i, j] = (codes[i, j].to(torch.float32) - 8.0) * scale[j // g, i] + zero[j // g, i]
    return t, w


def int8_tensor():
    n, k = 8, 16
    q = torch.tensor([[(i * 9 + j * 5) % 255 - 127 for j in range(k)] for i in range(n)], dtype=torch.int8)
    scale = torch.tensor([[0.25], [0.5], [0.125], [0.25], [0.5], [0.125], [0.25], [1.0]], dtype=torch.float32)
    t = Int8Tensor(q, scale, [1, k], torch.float32)
    return t, q.to(torch.float32) * scale


def float8_tensor():
    n, k = 8, 128
    values = torch.tensor([[(i * 11 + j * 3) % 448 - 224 for j in range(k)] for i in range(n)], dtype=torch.float32)
    q = (values / 8.0).to(torch.float8_e4m3fn)
    scale = torch.full((n, 1), 8.0, dtype=torch.float32)
    t = Float8Tensor(q, scale, block_size=[1, k], dtype=torch.float32)
    return t, q.to(torch.float32) * scale


def nvfp4_tensor():
    n, k = 32, 64
    hp = torch.tensor([[(i * 7 + j * 3) % 61 - 30 for j in range(k)] for i in range(n)], dtype=torch.float32) / 8.0
    per_tensor = torch.tensor(0.125, dtype=torch.float32)
    act = torch.tensor(0.5, dtype=torch.float32)
    t = NVFP4Tensor.to_nvfp4(hp, 16, per_tensor_scale=per_tensor, act_per_tensor_scale=act,
                             is_swizzled_scales=True)
    return t, hp


def mx_tensor(elem, swizzled):
    n, k = 32, 64
    hp = torch.tensor([[(i * 13 + j * 5) % 47 - 23 for j in range(k)] for i in range(n)], dtype=torch.float32) / 4.0
    t = MXTensor.to_mx(hp, elem, 32, is_swizzled_scales=swizzled)
    return t, hp


# --------------------------------------------------------------------------------------------- mslk preshuffle reference
#
# mslk's preshuffle_i4 runs three steps over an already quantized int4 weight: pack_int4 pairs each
# even column into the low nibble, CUTLASS reorder_tensor applies compute_memory_reordering_atom's
# permutation, and the FP8-activation path first encodes each code with unified_encode_int4b and
# afterwards expands each group scale into the eight-entry lookup of pack_scale_fp8. This module
# reproduces those steps directly; the permutation is checked against the pinned CUTLASS by
# ``reorder_layout`` below (same formulas as TorchaoSources.Preshuffled).

def reorder_positions(n, k, fp8):
    """The destination int4 position of each logical (row, column) of a preshuffled weight."""
    k_atom = 32 if fp8 else 16
    tile = 512 if fp8 else 256
    row_tiles = n // 16

    def position(r, c):
        if fp8:
            atom = 64 * (r % 8) + 4 * ((r // 8) % 2) + c % 4 + 16 * ((c // 4) % 4) + 8 * ((c % 32) // 16)
        else:
            atom = 32 * (r % 8) + ((r // 8) % 2) + 4 * (c % 2) + 8 * ((c // 2) % 4) + 2 * ((c // 8) % 2)
        return tile * ((r // 16) + row_tiles * (c // k_atom)) + atom

    return position


def encode_code(q):
    """unified_encode_int4b: the two's complement nibble of q, with +1..+7 mapped to 8-q."""
    if 1 <= q <= 7:
        return 8 - q
    return q & 0xF


def preshuffled_qdata(codes, fp8):
    """codes [N, K] (or [E, N, K]) in [-8, 7] -> the stored [N, K/2] (or [E, N, K/2]) uint8."""
    lead = codes.shape[:-2]
    n, k = codes.shape[-2], codes.shape[-1]
    experts = 1
    for d in lead:
        experts *= d
    position = reorder_positions(n, k, fp8)
    nibbles = torch.zeros(experts, n * k, dtype=torch.uint8)
    flat = codes.reshape(experts, n, k)
    for e in range(experts):
        for r in range(n):
            for c in range(k):
                q = int(flat[e, r, c].item())
                nibbles[e, position(r, c)] = encode_code(q) if fp8 else q & 0xF
    out = (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).to(torch.uint8)
    return out.reshape(lead + (n, k // 2))


def fp8_lookup_scales(scale):
    """The pack_scale_fp8 tensor of a [G, N] FP8 group scale: eight values per element, last = -scale."""
    g, n = scale.shape
    out = torch.zeros(g, 8, n, dtype=torch.uint8)
    flat = out.view(-1)
    for gi in range(g):
        for ni in range(n):
            s = float(scale[gi, ni].to(torch.float32))
            for j in range(8):
                value = torch.tensor(s * (-8 + j), dtype=torch.float32).to(torch.float8_e4m3fn)
                flat[(gi * n + ni) * 8 + j] = value.view(torch.uint8)
    return out.view(torch.float8_e4m3fn)


def preshuffle_bf16(experts=False):
    n, k, g = (16, 64, 32) if experts else (32, 64, 32)
    groups = k // g
    count = 2 if experts else 1
    shape = ((count,) if experts else ()) + (n, k)

    codes = torch.stack([
        torch.tensor([[(i * 5 + j * 3 + e * 2 + 1) % 16 - 8 for j in range(k)] for i in range(n)],
                     dtype=torch.int32)
        for e in range(count)]).reshape(shape)
    scale = torch.tensor([[0.25 if (gi + ni) % 2 == 0 else 0.5 for ni in range(n)] for gi in range(groups)],
                         dtype=torch.bfloat16)
    zero = torch.tensor([[0.0 if (gi + ni) % 3 == 0 else 0.25 for ni in range(n)] for gi in range(groups)],
                        dtype=torch.bfloat16)
    qdata = preshuffled_qdata(codes, False)
    if experts:
        scale = scale.unsqueeze(0).expand(count, -1, -1).contiguous()
        zero = zero.unsqueeze(0).expand(count, -1, -1).contiguous()
    block = [1] * (len(shape) - 1) + [g]
    t = Int4PreshuffledTensor(qdata=qdata, group_scale=scale, block_size=block, shape=torch.Size(shape),
                              group_zero=zero, row_scale=None)
    w = torch.empty(shape, dtype=torch.float32)
    for e in range(count):
        ce = codes if not experts else codes[e]
        se = scale if not experts else scale[e]
        ze = zero if not experts else zero[e]
        for i in range(n):
            for j in range(k):
                value = (ce[i, j].to(torch.float32) * float(se[j // g, i]) + float(ze[j // g, i]))
                if experts:
                    w[e, i, j] = value
                else:
                    w[i, j] = value
    return t, w


def preshuffle_fp8(experts=False):
    n, k, g = (16, 64, 32) if experts else (32, 64, 32)
    groups = k // g
    count = 2 if experts else 1
    shape = ((count,) if experts else ()) + (n, k)
    codes = torch.stack([
        torch.tensor([[(i * 7 + j * 3 + e * 2 + 1) % 16 - 8 for j in range(k)] for i in range(n)],
                     dtype=torch.int32) for e in range(count)]).reshape(shape)
    # The reference normalizes the group scale by a per-row maximum before casting it to E4M3.
    scale = torch.tensor([[1.0 if (gi + ni) % 2 == 0 else 0.5 for ni in range(n)] for gi in range(groups)],
                         dtype=torch.float32).to(torch.float8_e4m3fn)
    row_scale = torch.tensor([2.0 if ni % 2 == 0 else 4.0 for ni in range(n)], dtype=torch.float32)
    qdata = preshuffled_qdata(codes, True)
    packed = torch.stack([fp8_lookup_scales(scale) for _ in range(count)])
    if experts:
        scale = scale.unsqueeze(0).expand(count, -1, -1).contiguous()
        row_scale = row_scale.unsqueeze(0).expand(count, -1).contiguous()
    else:
        packed = packed[0]
    block = [1] * (len(shape) - 1) + [g]
    t = Int4PreshuffledTensor(qdata=qdata, group_scale=packed, block_size=block, shape=torch.Size(shape),
                              group_zero=None, row_scale=row_scale)
    w = torch.empty(shape, dtype=torch.float32)
    for e in range(count):
        ce = codes if not experts else codes[e]
        se = scale if not experts else scale[e]
        re = row_scale if not experts else row_scale[e]
        for i in range(n):
            for j in range(k):
                value = ce[i, j].to(torch.float32) * float(se[j // g, i].to(torch.float32)) * float(re[i])
                if experts:
                    w[e, i, j] = value
                else:
                    w[i, j] = value
    return t, w


def plain_int32_tensor():
    """XPU's Int4PlainInt32Tensor: unsigned 0..15 codes as [N, K/8] int32 words of four packed bytes."""
    n, k, g = 16, 64, 32
    groups = k // g
    codes = torch.tensor([[(i * 3 + j * 5 + 2) % 16 for j in range(k)] for i in range(n)], dtype=torch.int32)
    packed = (codes[:, ::2] | (codes[:, 1::2] << 4)).to(torch.uint8).contiguous()
    qdata = packed.view(torch.int32)
    scale = torch.tensor([[0.25 if (gi + ni) % 2 == 0 else 0.5 for ni in range(n)] for gi in range(groups)],
                         dtype=torch.bfloat16)
    zero = torch.tensor([[4 if (gi + ni) % 3 == 0 else 2 for ni in range(n)] for gi in range(groups)],
                        dtype=torch.int8)
    t = Int4PlainInt32Tensor(qdata, scale, zero, [1, g], torch.Size([n, k]))
    w = torch.empty(n, k, dtype=torch.float32)
    for i in range(n):
        for j in range(k):
            w[i, j] = (codes[i, j].to(torch.float32) - float(zero[j // g, i])) * float(scale[j // g, i])
    return t, w


def intx_tensor(bits):
    """An IntxUnpackedToInt8Tensor: one code per int8 byte, row-major [N, G] scale and zero point."""
    n, k, g = 16, 64, 32
    groups = k // g
    if bits == 4:
        codes = torch.tensor([[(i * 5 + j * 3 + 1) % 16 - 8 for j in range(k)] for i in range(n)],
                             dtype=torch.int8)
        zero_other = 1
    elif bits == 8:
        codes = torch.tensor([[(i * 37 + j * 11) % 255 - 127 for j in range(k)] for i in range(n)],
                             dtype=torch.int8)
        zero_other = -3
    else:
        lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        span = hi - lo + 1
        codes = torch.tensor([[(i * 5 + j * 3 + 1) % span + lo for j in range(k)] for i in range(n)],
                             dtype=torch.int8)
        # Symmetric: a zero point of 0 keeps the byte codes inside a Q8_0 block, so the int8 channel scheme
        # carries the int1..int7 codes losslessly.
        zero_other = 0
    scale = torch.tensor([[0.25 if (i + gi) % 2 == 0 else 0.5 for gi in range(groups)] for i in range(n)],
                         dtype=torch.float32)
    zero = torch.tensor([[0 if (i + gi) % 3 == 0 else zero_other for gi in range(groups)]
                         for i in range(n)], dtype=torch.int8)
    t = IntxUnpackedToInt8Tensor(codes, scale, zero, torch.int4 if bits == 4 else getattr(torch, "int%d" % bits),
                                 (1, g), torch.float32, None)
    w = torch.empty(n, k, dtype=torch.float32)
    for i in range(n):
        for j in range(k):
            w[i, j] = (codes[i, j].to(torch.float32) - float(zero[i, j // g])) * float(scale[i, j // g])
    return t, w


def intx_tensor_symmetric(bits):
    """The symmetric mapping: zero points at 0, which keeps the int8 codes inside a Q8_0 block."""
    n, k, g = 16, 64, 32
    groups = k // g
    codes = torch.tensor([[(i * 29 + j * 7) % 255 - 127 for j in range(k)] for i in range(n)], dtype=torch.int8)
    scale = torch.tensor([[0.25 if (i + gi) % 2 == 0 else 0.5 for gi in range(groups)] for i in range(n)],
                         dtype=torch.float32)
    zero = torch.zeros(n, groups, dtype=torch.int8)
    t = IntxUnpackedToInt8Tensor(codes, scale, zero, torch.int8 if bits == 8 else torch.int4, (1, g),
                                 torch.float32, None)
    w = codes.to(torch.float32) * scale.repeat_interleave(g, dim=1)
    return t, w


def int4_tile_experts():
    """A three-dimensional Int4TilePackedTo4dTensor: one tinygemm pack per expert."""
    experts, n, k, g, ntile = 2, 16, 128, 32, 8
    k_pad = ((k + 1023) // 1024) * 1024
    n_pad = ((n + ntile - 1) // ntile) * ntile
    stored_groups = k_pad // g
    codes = torch.zeros(experts, n_pad, k_pad, dtype=torch.uint8)
    for e in range(experts):
        codes[e, :n, :k] = torch.tensor([[(i * 3 + j * 7 + e * 5 + 2) & 0xF for j in range(k)] for i in range(n)],
                                        dtype=torch.uint8)
    packed = []
    for e in range(experts):
        inp = ((codes[e, :, ::2] << 4) | codes[e, :, 1::2]).to(torch.uint8).cuda()
        packed.append(torch.ops.aten._convert_weight_to_int4pack(inp.contiguous(), 8).cpu().unsqueeze(0))
    qdata = torch.cat(packed, dim=0)
    scale = torch.full((experts, n_pad, stored_groups), 0.5, dtype=torch.bfloat16)
    zero = torch.full((experts, n_pad, stored_groups), 0.25, dtype=torch.bfloat16)
    for e in range(experts):
        for c in range(stored_groups):
            for i in range(n):
                scale[e, i, c] = 0.25 if (c + i + e) % 3 == 0 else 0.5
                zero[e, i, c] = 0.0 if (c + i + e) % 2 == 0 else 0.5
    scale_and_zero = pack_tinygemm_scales_and_zeros(scale, zero, torch.bfloat16)
    t = Int4TilePackedTo4dTensor(qdata, scale_and_zero, [1, 1, g], torch.Size([experts, n, k]))
    w = torch.empty(experts, n, k, dtype=torch.float32)
    for e in range(experts):
        for i in range(n):
            for j in range(k):
                w[e, i, j] = ((codes[e, i, j].to(torch.float32) - 8.0)
                              * float(scale[e, i, j // g]) + float(zero[e, i, j // g]))
    return t, w


def int4_tile_ntile16():
    """The ROCm ntile 16 packing: N padded to 16, the [n/8, k/1024, 32, 4] layout unchanged."""
    n, k, g, ntile = 24, 128, 32, 16
    k_pad = ((k + 1023) // 1024) * 1024
    n_pad = ((n + ntile - 1) // ntile) * ntile
    stored_groups = k_pad // g
    codes = torch.zeros(n_pad, k_pad, dtype=torch.uint8)
    codes[:n, :k] = torch.tensor([[(i * 3 + j * 7 + 2) & 0xF for j in range(k)] for i in range(n)],
                                 dtype=torch.uint8)
    inp = ((codes[:, ::2] << 4) | codes[:, 1::2]).to(torch.uint8).cuda()
    qdata = torch.ops.aten._convert_weight_to_int4pack(inp.contiguous(), 8).cpu()
    scale = torch.full((n_pad, stored_groups), 0.5, dtype=torch.bfloat16)
    zero = torch.full((n_pad, stored_groups), 0.25, dtype=torch.bfloat16)
    for c in range(stored_groups):
        for i in range(n):
            scale[i, c] = 0.25 if (c + i) % 3 == 0 else 0.5
            zero[i, c] = 0.0 if (c + i) % 2 == 0 else 0.5
    scale_and_zero = pack_tinygemm_scales_and_zeros(scale, zero, torch.bfloat16)
    t = Int4TilePackedTo4dTensor(qdata, scale_and_zero, [1, g], torch.Size([n, k]))
    w = torch.empty(n, k, dtype=torch.float32)
    for i in range(n):
        for j in range(k):
            w[i, j] = (codes[i, j].to(torch.float32) - 8.0) * float(scale[i, j // g]) + float(zero[i, j // g])
    return t, w


def int4_pre_scale():
    """An Int4Tensor with an AWQ/SmoothQuant activation pre-scale: per input column K."""
    n, k, g = 8, 16, 8
    codes = torch.tensor([[(i * 5 + j * 3 + 1) & 0xF for j in range(k)] for i in range(n)], dtype=torch.uint8)
    signed = codes.to(torch.int8)
    signed[signed > 7] -= 16
    qdata = ((signed[:, ::2] & 0xF) | ((signed[:, 1::2] & 0xF) << 4)).to(torch.int8)
    rows = k // g
    scale = torch.tensor([[0.25, 0.5, 0.75, 0.25, 0.5, 0.75, 0.25, 0.5],
                          [0.5, 0.25, 0.5, 0.25, 0.75, 0.25, 0.5, 0.25]], dtype=torch.float32)
    zero = torch.tensor([[0.0, 0.25, 0.5, 0.0, 0.25, 0.5, 0.0, 0.25],
                         [0.5, 0.0, 0.25, 0.5, 0.0, 0.25, 0.5, 0.0]], dtype=torch.float32)
    pre = torch.tensor([0.5 if j % 2 == 0 else 2.0 for j in range(k)], dtype=torch.float32)
    t = Int4Tensor(qdata, scale, zero, [1, g], torch.Size([n, k]), act_pre_scale=pre)
    w = torch.empty(n, k, dtype=torch.float32)
    for i in range(n):
        for j in range(k):
            w[i, j] = (signed[i, j].to(torch.float32) * scale[j // g, i] + zero[j // g, i]) * float(pre[j])
    return t, w


def int8_pre_scale():
    n, k = 8, 16
    q = torch.tensor([[(i * 9 + j * 5) % 255 - 127 for j in range(k)] for i in range(n)], dtype=torch.int8)
    scale = torch.tensor([[0.25], [0.5], [0.125], [0.25], [0.5], [0.125], [0.25], [1.0]], dtype=torch.float32)
    pre = torch.tensor([0.5 if j % 2 == 0 else 2.0 for j in range(k)], dtype=torch.float32)
    t = Int8Tensor(q, scale, [1, k], torch.float32, act_pre_scale=pre)
    return t, q.to(torch.float32) * scale * pre


def int8_static_activation():
    """Static activation quantization: the weight is plain int8, with the activation's qparams siblings."""
    n, k = 8, 16
    q = torch.tensor([[(i * 7 + j * 3) % 255 - 127 for j in range(k)] for i in range(n)], dtype=torch.int8)
    scale = torch.tensor([[0.25], [0.5], [0.125], [0.25], [0.5], [0.125], [0.25], [1.0]], dtype=torch.float32)
    act_scale = torch.tensor([0.5], dtype=torch.float32)
    act_zero = torch.tensor([5], dtype=torch.int8)
    kwargs = _int8_kwargs()
    t = Int8Tensor(q, scale, [1, k], torch.float32, act_quant_scale=act_scale,
                   act_quant_zero_point=act_zero, act_quant_kwargs=kwargs)
    return t, q.to(torch.float32) * scale


def _int8_kwargs():
    from torchao.quantization.quant_api import QuantizeTensorToInt8Kwargs
    from torchao.quantization.granularity import PerRow
    from torchao.quantization.quant_primitives import MappingType
    return QuantizeTensorToInt8Kwargs(granularity=PerRow(), mapping_type=MappingType.ASYMMETRIC)


def int8_pre_scale_with_activation():
    """An activation pre-scale and static activation quantization together: the weight alone, the scale as runtime."""
    n, k = 8, 16
    q = torch.tensor([[(i * 7 + j * 3) % 255 - 127 for j in range(k)] for i in range(n)], dtype=torch.int8)
    scale = torch.tensor([[0.25], [0.5], [0.125], [0.25], [0.5], [0.125], [0.25], [1.0]], dtype=torch.float32)
    pre = torch.tensor([0.5 if j % 2 == 0 else 2.0 for j in range(k)], dtype=torch.float32)
    t = Int8Tensor(q, scale, [1, k], torch.float32, act_quant_scale=torch.tensor([0.5], dtype=torch.float32),
                   act_quant_zero_point=torch.tensor([5], dtype=torch.int8), act_pre_scale=pre,
                   act_quant_kwargs=_int8_kwargs())
    return t, q.to(torch.float32) * scale


# --------------------------------------------------------------------------------------------- canonical tensor reference

def suffix(name, s):
    return name[:-len(".weight")] + "." + s if name.endswith(".weight") else name + "_" + s


def decode_preshuffled(qdata, n, k, fp8):
    """The stored bytes as the plain two's complement nibbles the loader exposes."""
    lead = qdata.shape[:-2]
    experts = 1
    for d in lead:
        experts *= d
    flat = qdata.reshape(experts, n * (k // 2))
    position = reorder_positions(n, k, fp8)
    nibbles = torch.zeros(experts, n, k, dtype=torch.uint8)
    for e in range(experts):
        for r in range(n):
            for c in range(k):
                at = position(r, c)
                byte = int(flat[e, at // 2].item()) & 0xFF
                code = (byte >> (4 * (at % 2))) & 0xF
                if fp8:
                    if code == 0:
                        q = 0
                    elif code < 8:
                        q = 8 - code
                    else:
                        q = code - 16
                else:
                    q = code - 16 if code >= 8 else code
                nibbles[e, r, c] = q & 0xF
    packed = (nibbles[:, :, 0::2] | (nibbles[:, :, 1::2] << 4)).to(torch.uint8)
    return packed.reshape(lead + (n, k // 2))


def lookup_scale(packed, groups, n, row_scale):
    """The logical group scale of an FP8-activation preshuffle: row_scale[n] * -lookup[7]."""
    lead = packed.shape[:-3] if packed.dim() >= 3 else ()
    experts = 1
    for d in lead:
        experts *= int(d)
    raw = packed.reshape(experts, groups, 8, n).contiguous().view(torch.uint8).reshape(-1)
    picked = torch.zeros(experts, groups, n, dtype=torch.uint8)
    for e in range(experts):
        for gi in range(groups):
            for ni in range(n):
                picked[e, gi, ni] = raw[((e * groups + gi) * n + ni) * 8 + 7]
    scale = -picked.view(torch.float8_e4m3fn).to(torch.float32)
    row = row_scale.reshape(experts, n).to(torch.float32)
    out = torch.empty(experts, groups, n, dtype=torch.float32)
    for e in range(experts):
        for gi in range(groups):
            for ni in range(n):
                out[e, gi, ni] = scale[e, gi, ni] * row[e, ni]
    return out.reshape(lead + (groups, n))


def packed_nibbles(values):
    """Signed (or unsigned) int4 values as the plain two-nibble bytes, element 2j in the low nibble."""
    v = values & 0xF
    return (v[..., 0::2] | (v[..., 1::2] << 4)).to(torch.uint8)


def canonical(t, name):
    """The loader's decoded tensors: name -> (tensor). Mirrors TorchaoDecoder."""
    out = {}
    cls = t.__class__.__name__
    if cls == "Int4Tensor":
        out[name] = t.qdata
        out[suffix(name, "weight_scale")] = t.scale
        out[suffix(name, "weight_zero_point")] = t.zero_point
    elif cls == "Int4TilePackedTo4dTensor":
        shape = [int(d) for d in t.shape]
        n, k = shape[-2], shape[-1]
        g = int(t.block_size[-1])
        groups = (k + g - 1) // g
        flat = detile(t.qdata, n, k)
        out[name] = flat
        out[suffix(name, "weight_scale")] = t.scale_and_zero[..., :groups, :n, 0].contiguous()
        out[suffix(name, "weight_zero_point")] = t.scale_and_zero[..., :groups, :n, 1].contiguous()
    elif cls == "Int4PreshuffledTensor":
        shape = [int(d) for d in t.shape]
        n, k = shape[-2], shape[-1]
        fp8 = t.row_scale is not None
        out[name] = decode_preshuffled(t.qdata, n, k, fp8)
        if fp8:
            groups = k // int(t.block_size[-1])
            out[suffix(name, "weight_scale")] = lookup_scale(t.group_scale, groups, n, t.row_scale)
        else:
            out[suffix(name, "weight_scale")] = t.group_scale
            out[suffix(name, "weight_zero_point")] = t.group_zero
    elif cls == "Int4PlainInt32Tensor":
        shape = [int(d) for d in t.shape]
        n, k = shape[-2], shape[-1]
        out[name] = t.qdata.view(torch.uint8).reshape(n, k // 2) ^ 0x88
        out[suffix(name, "weight_scale")] = t.scale
        groups = k // int(t.block_size[-1])
        out[suffix(name, "weight_zero_point")] = (8.0 - t.zero_point.to(torch.float32)) * t.scale.to(torch.float32)
    elif cls == "IntxUnpackedToInt8Tensor":
        shape = [int(d) for d in t.shape]
        n, k = shape[-2], shape[-1]
        bits = str(t.target_dtype).split(".")[-1]
        groups = k // int(t.block_size[-1])
        if bits == "int4":
            out[name] = packed_nibbles(t.qdata)
            out[suffix(name, "weight_scale")] = t.scale.t().contiguous()
            out[suffix(name, "weight_zero_point")] = (-t.zero_point.to(torch.float32) * t.scale.to(torch.float32)).t().contiguous()
        else:
            out[name] = t.qdata
            out[suffix(name, "weight_scale")] = t.scale
            out[suffix(name, "weight_zero_point")] = t.zero_point
    elif cls == "Int8Tensor":
        out[name] = t.qdata
        out[suffix(name, "weight_scale")] = t.scale
        if t.zero_point is not None:
            out[suffix(name, "weight_zero_point")] = t.zero_point
        if t.act_quant_scale is not None:
            out[suffix(name, "input_scale")] = t.act_quant_scale
        if t.act_quant_zero_point is not None:
            out[suffix(name, "input_zero_point")] = t.act_quant_zero_point
    elif cls == "Float8Tensor":
        out[name] = t.qdata
        out[suffix(name, "weight_scale")] = t.scale
    elif cls == "MXTensor":
        elem = str(t.elem_dtype).split(".")[-1]
        out[suffix(name, "weight_packed" if elem == "float4_e2m1fn_x2" else "weight")] = t.qdata
        groups = int(t.qdata.shape[-1]) * (2 if elem == "float4_e2m1fn_x2" else 1) // int(t.block_size)
        out[suffix(name, "weight_scale")] = unswizzle(t.scale, t.is_swizzled_scales, int(t.qdata.shape[-2]),
                                                       groups)
    elif cls == "NVFP4Tensor":
        out[name] = t.qdata
        groups = int(t.qdata.shape[-1]) * 2 // 16
        out[suffix(name, "weight_scale")] = unswizzle(t.scale, t.is_swizzled_scales, int(t.qdata.shape[-2]),
                                                      groups)
        out[suffix(name, "weight_scale_2")] = (t.per_tensor_scale if t.per_tensor_scale is not None
                                               else torch.ones(1, dtype=torch.float32))
        if t.act_per_tensor_scale is not None:
            out[suffix(name, "input_scale")] = t.act_per_tensor_scale
    else:
        raise ValueError(cls)
    return out


def detile(qdata, n, k):
    shape = [int(d) for d in qdata.shape]
    half = shape[-1]                  # inner_k_tiles / 2
    k_blocks = shape[-3]
    row_blocks = shape[-4]
    experts = 1
    for d in shape[:-4]:
        experts *= d
    row_bytes = half * 4
    k_block_bytes = half * 16
    b = raw(qdata)
    out = torch.empty(experts, n, k // 2, dtype=torch.uint8)
    for e in range(experts):
        for r in range(n):
            for c in range(k // 2):
                block = r // 8
                r_in = r % 8
                k_block = c // k_block_bytes
                tile = (c % k_block_bytes) // 16
                c4 = c % 4
                j = (c % 16) // 4
                row = ((e * row_blocks + block) * k_blocks + k_block) * 32 + 4 * r_in + c4
                at = row * row_bytes + tile * 4
                w = struct.unpack_from("<I", b, at)[0]
                even = ((w >> (4 * j)) & 0xF) ^ 0x8
                odd = ((w >> (4 * (4 + j))) & 0xF) ^ 0x8
                out[e, r, c] = even | (odd << 4)
    return out.reshape(*shape[:-4], n, k // 2)


def unswizzle(scale, swizzled, rows, groups):
    if not swizzled:
        return scale.contiguous()
    src = scale.reshape(-1).view(torch.uint8).tolist()
    row_blocks = (rows + 127) // 128
    col_blocks = (groups + 3) // 4
    dtype = scale.dtype
    vals = torch.empty(rows, groups, dtype=dtype)
    dst = vals.view(torch.uint8).reshape(-1)
    for r in range(rows):
        for c in range(groups):
            at = ((r // 128) * col_blocks + c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4
            dst[r * groups + c] = src[at]
    return vals


# --------------------------------------------------------------------------------------------- fixtures

def flatten_preshuffle(name, t, bias):
    """The flatten_tensor_state_dict form for Int4PreshuffledTensor, which torchao's list omits."""
    module, weight = name.rsplit(".", 1)
    prefix = module + "._" + weight + "_"
    data = {}
    for field in ("qdata", "group_scale", "group_zero", "row_scale"):
        value = getattr(t, field)
        if value is not None:
            data[field] = value
    tensors = {prefix + field: value for field, value in data.items()}
    tensors[BIAS] = bias
    metadata = {
        "tensor_names": json.dumps([name, BIAS]),
        name: json.dumps({
            "_type": t.__class__.__name__,
            "_data": {"block_size": [int(x) for x in t.block_size], "shape": [int(x) for x in t.shape]},
            "_tensor_data_names": list(data),
        }),
        BIAS: json.dumps({"_type": "Tensor"}),
    }
    return tensors, metadata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=FIXTURES)
    ap.add_argument("--only", default="",
                    help="comma-separated fixture names; the reference file keeps its other entries")
    args = ap.parse_args()
    out = args.out
    os.makedirs(out, exist_ok=True)
    reference = os.path.join(out, "torchao-reference.json")
    ref = {}
    if args.only and os.path.exists(reference):
        with open(reference) as f:
            ref = json.load(f)

    def record(name, weight, dequant, extra=None):
        c = canonical(weight, W)
        entry = {
            "canonical": {k: {"dtype": dtype_name(v), "shape": npy(v), "sha256": digest(v)} for k, v in c.items()},
            "dequantized": f32_digest(dequant),
        }
        pre = getattr(weight, "act_pre_scale", None)
        if pre is not None:
            entry["activation_pre_scale"] = [float(v) for v in pre.reshape(-1).tolist()]
        if extra:
            entry.update(extra)
        ref[name] = entry

    cases = []
    t, signed, scale, zero = int4_tensor()
    w = torch.empty(8, 16, dtype=torch.float32)
    for i in range(8):
        for j in range(16):
            w[i, j] = signed[i, j].to(torch.float32) * scale[j // 8, i] + zero[j // 8, i]
    cases.append(("int4", t, w))
    t, w = int4_tile_tensor()
    cases.append(("int4-tile", t, w))
    for name, make in (("int8", int8_tensor), ("float8", float8_tensor), ("nvfp4", nvfp4_tensor)):
        t, fallback = make()
        cases.append((name, t, t.dequantize(torch.float32) if hasattr(t, "dequantize") else fallback))
    for name, elem, swizzled in (("mxfp4", torch.float4_e2m1fn_x2, True),
                                 ("mxfp8", torch.float8_e4m3fn, False)):
        t, fallback = mx_tensor(elem, swizzled)
        cases.append((name, t, t.dequantize(torch.float32) if hasattr(t, "dequantize") else fallback))

    cases.append(("preshuffle-bf16", *preshuffle_bf16()))
    cases.append(("preshuffle-bf16-expert", *preshuffle_bf16(experts=True)))
    cases.append(("preshuffle-fp8", *preshuffle_fp8()))
    cases.append(("preshuffle-fp8-expert", *preshuffle_fp8(experts=True)))
    cases.append(("plain-int32", *plain_int32_tensor()))
    cases.append(("intx4", *intx_tensor(4)))
    for bits in (1, 2, 3, 5, 6, 7, 8):
        cases.append(("intx%d" % bits, *intx_tensor(bits)))
    cases.append(("intx8-sym", *intx_tensor_symmetric(8)))
    cases.append(("int4-tile-expert", *int4_tile_experts()))
    cases.append(("int4-tile-ntile16", *int4_tile_ntile16()))
    cases.append(("int4-pre-scale", *int4_pre_scale()))
    cases.append(("int8-pre-scale", *int8_pre_scale()))
    cases.append(("int8-static-act", *int8_static_activation()))
    cases.append(("int8-pre-scale-act", *int8_pre_scale_with_activation()))

    wanted = [n for n in args.only.split(",") if n] or None
    if wanted is not None:
        unknown = set(wanted) - {c[0] for c in cases}
        if unknown:
            raise SystemExit("unknown fixtures: " + ", ".join(sorted(unknown)))
        cases = [c for c in cases if c[0] in wanted]

    bias = torch.arange(8, dtype=torch.float32)
    for name, t, w in cases:
        if t.__class__.__name__ == "Int4PreshuffledTensor":
            tensors, metadata = flatten_preshuffle(W, t, bias)
        else:
            tensors, metadata = flatten_tensor_state_dict({W: t, BIAS: bias})
        d = os.path.join(out, name)
        write_safetensors(os.path.join(d, "model.safetensors"), tensors, metadata)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump({"architectures": ["A"]}, f)
        record(name, t, w)

    # Pickles: torch.save on CPU writes the wrapper-subclass form and, for FP8, the v3 views.
    if wanted is None or "int4" in wanted or "float8" in wanted:
        for name in ("int4", "float8"):
            if wanted is not None and name not in wanted:
                continue
            t = int4_tensor()[0] if name == "int4" else float8_tensor()[0]
            case = next(c for c in cases if c[0] == name)
            d = os.path.join(out, name + "-pickle")
            os.makedirs(d, exist_ok=True)
            torch.save({W: t, BIAS: bias}, os.path.join(d, "pytorch_model.bin"))
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump({"architectures": ["A"]}, f)
            record(name + "-pickle", t, case[2])

    with open(reference, "w") as f:
        json.dump(ref, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
