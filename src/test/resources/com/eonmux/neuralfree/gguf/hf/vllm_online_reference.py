"""The reference codes of vLLM's online FP8 quantization (commit 8530a778f) for vllm-online-layouts.txt.

A NumPy restatement of vLLM's FP8 online quantizers, evaluated in float32 as PyTorch evaluates them:
  fp8_per_tensor   scale = amax / 448 (true division, online/fp8.py _fp8_scale) over the fused weight, per expert
                   for routed experts; codes e4m3(clamp(x * (1 / scale), +-448)) (scaled_fp8_quant, static scale).
  fp8_per_block    per_block_cast_to_fp8 (utils/deep_gemm.py): the fused weight zero-padded to whole 128x128 tiles,
                   scale = max(amax, 1e-4) * (1 / 448) per tile, codes e4m3(x * (1 / scale)).
  fp8_per_channel  scale = max(amax / 448, 1 / (448 * 512)) per row (_fp8_channel_scale), codes
                   e4m3(clamp(x / scale, +-448)) (_fp8_quant_per_channel; the routed w13's dynamic per-token
                   kernel computes the same).
e4m3 is round-to-nearest-even to float8_e4m3fn. The fused weights are vLLM's: q|k|v, gate|up (the routed experts'
w13 with each shard padded to 128 rows), MLA's fused_qkv_a_proj (q_a|kv_a_proj_with_mqa) and kv_b_proj, Qwen3.5's
in_proj_qkvz (in_proj_qkv|in_proj_z) and in_proj_ba (in_proj_b|in_proj_a), all in checkpoint row order. Each
quantized weight is then laid out as NeuralFree stores it (the LLaMA q/k rope permutation, MLA's per-head k_b
transpose and v_b split, Qwen3.5's value-head regrouping), and digested as HfOnlineQuantParityTest digests:
  codes   SHA-256 of one E4M3 byte per stored element, row-major
  scales  SHA-256 of the f32 scale (little-endian) of each stored block (128 values, or 32 for rows that are no
          whole 128-value blocks), row by row
  values  SHA-256 of the stored BF16 bytes (MLA's k_b under a per-row method: vLLM's dequantized W_UK)
--check verifies the restatement against the fp8 digests vLLM itself produced (vllm-online-parity.txt).

Usage: python3 vllm_online_reference.py [--check vllm-online-parity.txt]
"""
import hashlib
import sys

import numpy as np

F32 = np.float32
FP8_MAX = F32(448.0)


# ---------------------------------------------------------------------------------------------- inputs

def lcg_bf16(rows, cols, seed, base):
    """The tests' BF16 weight (HfOnlineQuantParityTest.lcgBf16), as float32."""
    out = np.empty(rows * cols, dtype=np.uint32)
    s = seed & 0xFFFFFFFF
    for i in range(rows * cols):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        a = s
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        b = s
        e = (a >> 24) & 7
        e = e - 2 if e >= 6 else e
        if (b >> 22) == 0:
            e = 6 + ((b >> 21) & 1)
        out[i] = ((a >> 31) << 15) | ((base + e) << 7) | ((a >> 8) & 0x7F)
    return (out << 16).view(np.float32).reshape(rows, cols)


# ---------------------------------------------------------------------------------------------- e4m3

def _e4m3_table():
    vals = []
    for code in range(0x7F):
        e, m = code >> 3, code & 7
        v = (m / 8.0) * 2.0 ** -6 if e == 0 else (1 + m / 8.0) * 2.0 ** (e - 7)
        vals.append(v)
    return np.array(vals, dtype=np.float64)


E4M3 = _e4m3_table()


def e4m3(x):
    """float8_e4m3fn bytes of float32 x: nearest, ties to even; NaN beyond the 464 rounding bound."""
    x = np.asarray(x, dtype=np.float32)
    a = np.abs(x).astype(np.float64).ravel()
    hi = np.clip(np.searchsorted(E4M3, a, side='left'), 0, len(E4M3) - 1)
    lo = np.clip(hi - 1, 0, len(E4M3) - 1)
    dl, dh = a - E4M3[lo], E4M3[hi] - a
    pick = np.where(dl < dh, lo, np.where(dh < dl, hi, np.where(lo % 2 == 0, lo, hi)))
    pick = np.where(a >= 464.0, 0x7F, pick).astype(np.uint8)
    sign = (np.signbit(x.ravel())).astype(np.uint8) << 7
    return (pick | sign).reshape(x.shape)


def e4m3_value(codes):
    c = codes.astype(np.int64)
    v = E4M3[np.minimum(c & 0x7F, 0x7E)].astype(np.float32)
    return np.where(c & 0x80, -v, v).astype(np.float32)


# ---------------------------------------------------------------------------------------------- vLLM

def per_tensor(w):
    """codes [rows, cols], scale per element (one value) of one 2-D weight."""
    amax = F32(np.max(np.abs(w)))
    scale = F32(amax / FP8_MAX)
    inv = F32(F32(1.0) / scale)
    q = e4m3(np.clip(w * inv, -FP8_MAX, FP8_MAX).astype(np.float32))
    return q, np.full(w.shape, scale, dtype=np.float32)


def per_block(w):
    rows, cols = w.shape
    pr, pc = -(-rows // 128) * 128, -(-cols // 128) * 128
    p = np.zeros((pr, pc), dtype=np.float32)
    p[:rows, :cols] = w
    t = p.reshape(pr // 128, 128, pc // 128, 128)
    amax = np.maximum(np.abs(t).max(axis=(1, 3), keepdims=True), F32(1e-4)).astype(np.float32)
    sf = (amax * F32(1.0 / 448.0)).astype(np.float32)
    q = e4m3((t * (F32(1.0) / sf)).astype(np.float32)).reshape(pr, pc)[:rows, :cols]
    s = np.broadcast_to(sf, t.shape).reshape(pr, pc)[:rows, :cols]
    return q, np.ascontiguousarray(s)


def per_channel(w):
    amax = np.abs(w).max(axis=1, keepdims=True).astype(np.float32)
    scale = np.maximum((amax / FP8_MAX).astype(np.float32), F32(1.0) / (FP8_MAX * F32(512))).astype(np.float32)
    q = e4m3(np.clip((w / scale).astype(np.float32), -FP8_MAX, FP8_MAX).astype(np.float32))
    return q, np.broadcast_to(scale, w.shape).astype(np.float32)


METHODS = {'fp8_per_tensor': per_tensor, 'fp8_per_block': per_block, 'fp8_per_channel': per_channel}


def fused(method, parts, pad128=False):
    """vLLM's fused weight of parts (2-D, checkpoint order) quantized: per part, (codes, per-element scales)."""
    spans, at = [], 0
    for p in parts:
        spans.append((at, p.shape[0]))
        at += -(-p.shape[0] // 128) * 128 if pad128 else p.shape[0]
    w = np.zeros((at, parts[0].shape[1]), dtype=np.float32)
    for (o, n), p in zip(spans, parts):
        w[o:o + n] = p
    if method == 'fp8_per_tensor' and pad128:
        q, s = per_tensor(w)
    else:
        q, s = METHODS[method](w)
    return [(q[o:o + n], s[o:o + n]) for o, n in spans]


# ---------------------------------------------------------------------------------------------- digests

def sha(b):
    return hashlib.sha256(b).hexdigest()


def digests(codes, scales):
    """codes/scales [rows, cols] (stacked experts flattened): the codes and the per-block scales."""
    cols = codes.shape[-1]
    block = 128 if cols % 128 == 0 else 32
    c = codes.reshape(-1, cols)
    s = scales.reshape(-1, cols)[:, ::block]
    return {'codes': sha(np.ascontiguousarray(c).tobytes()),
            'scales': sha(np.ascontiguousarray(s).astype('<f4').tobytes())}


def bf16_bytes(x):
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    return r.astype('<u2').tobytes()


# ---------------------------------------------------------------------------------------------- layouts

def llama_permute(w, n_head):
    """llama.cpp LlamaModel.permute over rows of a [rows, ...] array."""
    shape = w.shape
    return w.reshape(n_head, 2, shape[0] // n_head // 2, *shape[1:]).swapaxes(1, 2).reshape(shape)


def value_heads(w, from_row, n_k, per_k, head_dim):
    """Qwen3.5 _reorder_v_heads on rows from from_row on: tiled unit v * nK + k is grouped unit k * perK + v."""
    out = w.copy()
    n_v = n_k * per_k
    for u in range(n_v):
        src = (u % n_k) * per_k + u // n_k
        out[from_row + u * head_dim:from_row + (u + 1) * head_dim] = \
            w[from_row + src * head_dim:from_row + (src + 1) * head_dim]
    return out


# ---------------------------------------------------------------------------------------------- scenarios

def qwen2_moe():
    """HfOnlineQuantParityTest's checkpoint: per method, GGUF name -> (codes, scales) as stored."""
    W = lambda r, c, seed, base: lcg_bf16(r, c, seed, base)
    q, k, v = W(256, 256, 11, 118), W(64, 256, 12, 121), W(64, 256, 13, 116)
    o = W(256, 256, 14, 119)
    ex = [[W(192, 256, 21, 117), W(192, 256, 22, 120), W(256, 192, 23, 117)],
          [W(192, 256, 24, 121), W(192, 256, 25, 118), W(256, 192, 26, 119)]]
    sg, su, sd = W(160, 256, 31, 117), W(160, 256, 32, 120), W(256, 160, 33, 118)
    out = {}
    for m in METHODS:
        t = {}
        (t['attn_q'], t['attn_k'], t['attn_v']) = fused(m, [q, k, v])
        t['attn_output'] = fused(m, [o])[0]
        (t['ffn_gate_shexp'], t['ffn_up_shexp']) = fused(m, [sg, su])
        t['ffn_down_shexp'] = fused(m, [sd])[0]
        gate, up, down = [], [], []
        for e in ex:
            g, u = fused(m, [e[0], e[1]], pad128=True)
            gate.append(g)
            up.append(u)
            down.append(fused(m, [e[2]])[0])
        for name, parts in (('ffn_gate_exps', gate), ('ffn_up_exps', up), ('ffn_down_exps', down)):
            t[name] = (np.stack([p[0] for p in parts]), np.stack([p[1] for p in parts]))
        out[m] = t
    return out


def llama():
    """LlamaForCausalLM: hidden 384, 4 heads of 96 (heads straddle 128-row tiles), 2 KV heads, ffn 320."""
    W = lambda r, c, seed, base: lcg_bf16(r, c, seed, base)
    q, k, v = W(384, 384, 41, 118), W(192, 384, 42, 121), W(192, 384, 43, 116)
    o = W(384, 384, 44, 119)
    g, u, d = W(320, 384, 45, 117), W(320, 384, 46, 120), W(384, 320, 47, 118)
    out = {}
    for m in ('fp8_per_tensor', 'fp8_per_block'):
        t = {}
        (qq, qs), (kq, ks), t['attn_v'] = fused(m, [q, k, v])
        t['attn_q'] = (llama_permute(qq, 4), llama_permute(qs, 4))
        t['attn_k'] = (llama_permute(kq, 2), llama_permute(ks, 2))
        t['attn_output'] = fused(m, [o])[0]
        (t['ffn_gate'], t['ffn_up']) = fused(m, [g, u])
        t['ffn_down'] = fused(m, [d])[0]
        out[m] = t
    return out


def deepseek_v2():
    """DeepseekV2ForCausalLM: hidden 256, 2 heads, q_lora 192, kv_lora 160, nope 64, rope 32, v 64, ffn 256."""
    W = lambda r, c, seed, base: lcg_bf16(r, c, seed, base)
    heads, nope, rope, vd, kvl = 2, 64, 32, 64, 160
    qa, qb = W(192, 256, 51, 118), W(heads * (nope + rope), 192, 52, 120)
    kva, kvb = W(kvl + rope, 256, 53, 117), W(heads * (nope + vd), kvl, 54, 121)
    o = W(256, heads * vd, 55, 119)
    g, u, d = W(256, 256, 56, 117), W(256, 256, 57, 120), W(256, 256, 58, 118)
    out = {}
    for m in METHODS:
        t = {}
        (t['attn_q_a'], t['attn_kv_a_mqa']) = fused(m, [qa, kva])
        t['attn_q_b'] = fused(m, [qb])[0]
        cq, cs = fused(m, [kvb])[0]
        cq3 = cq.reshape(heads, nope + vd, kvl)
        cs3 = cs.reshape(heads, nope + vd, kvl)
        t['attn_v_b'] = (cq3[:, nope:, :], cs3[:, nope:, :])
        kq, ks = cq3[:, :nope, :].transpose(0, 2, 1), cs3[:, :nope, :].transpose(0, 2, 1)
        if m == 'fp8_per_channel':
            t['attn_k_b values'] = bf16_bytes((e4m3_value(kq) * ks).astype(np.float32))
        else:
            t['attn_k_b'] = (np.ascontiguousarray(kq), np.ascontiguousarray(ks))
        t['attn_output'] = fused(m, [o])[0]
        (t['ffn_gate'], t['ffn_up']) = fused(m, [g, u])
        t['ffn_down'] = fused(m, [d])[0]
        out[m] = t
    return out


def qwen35():
    """Qwen3_5ForConditionalGeneration's linear-attention layer: hidden 256, 2 key heads of 32, 4 value heads of
    48 (the regrouped value heads straddle 128-row tiles)."""
    W = lambda r, c, seed, base: lcg_bf16(r, c, seed, base)
    n_k, n_v, hk, hv = 2, 4, 32, 48
    qkv, z = W(2 * n_k * hk + n_v * hv, 256, 61, 118), W(n_v * hv, 256, 62, 120)
    b, a = W(n_v, 256, 63, 117), W(n_v, 256, 64, 121)
    out = {}
    for m in ('fp8_per_tensor', 'fp8_per_block'):
        t = {}
        (cq, sq), (cz, sz) = fused(m, [qkv, z])
        t['attn_qkv'] = (value_heads(cq, 2 * n_k * hk, n_k, n_v // n_k, hv),
                         value_heads(sq, 2 * n_k * hk, n_k, n_v // n_k, hv))
        t['attn_gate'] = (value_heads(cz, 0, n_k, n_v // n_k, hv), value_heads(sz, 0, n_k, n_v // n_k, hv))
        (cb, sb), (ca, sa) = fused(m, [b, a])
        t['ssm_beta'] = (value_heads(cb, 0, n_k, n_v // n_k, 1), value_heads(sb, 0, n_k, n_v // n_k, 1))
        t['ssm_alpha'] = (value_heads(ca, 0, n_k, n_v // n_k, 1), value_heads(sa, 0, n_k, n_v // n_k, 1))
        out[m] = t
    return out


SCENARIOS = (('qwen2_moe', qwen2_moe), ('llama', llama), ('deepseek_v2', deepseek_v2), ('qwen35', qwen35))


def lines(name, table):
    for m, tensors in table.items():
        for tensor, value in tensors.items():
            if tensor.endswith(' values'):
                yield '%s %s blk.0.%s.weight values %s' % (name, m, tensor.split()[0], sha(value))
                continue
            for kind, digest in digests(*value).items():
                yield '%s %s blk.0.%s.weight %s %s' % (name, m, tensor, kind, digest)


def check(path):
    """The restatement against vLLM's own fp8 digests of the Qwen2-MoE checkpoint."""
    ours = {}
    for line in lines('qwen2_moe', qwen2_moe()):
        _, m, tensor, kind, digest = line.split()
        ours[(m, tensor, kind)] = digest
    bad = checked = 0
    for line in open(path, encoding='utf-8'):
        if line.startswith('#') or not line.strip():
            continue
        m, tensor, kind, digest = line.split(' ', 3)
        if m not in ('fp8_per_tensor', 'fp8_per_block'):
            continue
        checked += 1
        if ours.get((m, tensor, kind)) != digest.strip():
            bad += 1
            print('MISMATCH', m, tensor, kind)
    print('checked %d vLLM digests, %d mismatches' % (checked, bad))
    return bad == 0 and checked > 0


def main(argv):
    if len(argv) == 3 and argv[1] == '--check':
        return 0 if check(argv[2]) else 1
    for name, fn in SCENARIOS:
        for line in lines(name, fn()):
            print(line)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
