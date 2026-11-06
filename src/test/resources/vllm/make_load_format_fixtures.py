#!/usr/bin/env python3
"""Generates the vLLM load-format fixtures under src/test/resources/vllm/fixtures.

The fixtures are small, deterministic checkpoints written exactly in the on-disk formats:

  * a Hugging Face safetensors checkpoint (config.json, tokenizer files, model.safetensors);
  * vLLM's sharded_state layout (ShardedStateLoader.save_model): per-rank
    model-rank-<rank>-part-<part>.safetensors files of the vLLM module state dict, with the
    rank-local packing of QKVParallelLinear/MergedColumnParallelLinear/FusedMoE, the padded
    vocabulary and vLLM's tied-weight filtering;
  * CoreWeave tensorizer files (TensorSerializer): magic, data version, file header, 256 KiB
    metadata index and sequential per-tensor headers; encrypted files use data version 3 and
    libsodium's crypto_secretbox_detached (XSalsa20-Poly1305).

Every checkpoint is produced from the same tensors as its Hugging Face twin, so tests can load
both and compare. Run with a Python that has PyNaCl:

    python src/test/resources/vllm/make_load_format_fixtures.py
"""

import hashlib
import json
import os
import struct
import sys

from nacl.bindings import crypto_secretbox_easy

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fixtures")

TOKENIZER = {
    "added_tokens": [
        {"id": 0, "content": "<s>", "special": True, "normalized": False},
        {"id": 1, "content": "</s>", "special": True, "normalized": False},
        {"id": 5, "content": "<|begin|>", "special": True, "normalized": False},
        {"id": 6, "content": "Ġhi", "special": False, "normalized": False, "lstrip": True},
    ],
    "normalizer": None,
    "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True,
                      "use_regex": True},
    "post_processor": None,
    "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True,
                "use_regex": True},
    "model": {"type": "BPE", "unk_token": None, "byte_fallback": False, "ignore_merges": False,
              "vocab": {"<s>": 0, "</s>": 1, "a": 2, "b": 3, "ab": 4}, "merges": ["a b"]},
}
TOKENIZER_CONFIG = {"bos_token": "<s>", "eos_token": "</s>", "add_bos_token": False,
                    "clean_up_tokenization_spaces": False}


# ----------------------------------------------------------------------------- tensors

class Tensors:
    """Ordered tensors of one checkpoint, values seeded by insertion position (as SyntheticHf)."""

    def __init__(self):
        self.order = []
        self.data = {}
        self.shape = {}

    def add(self, name, shape, values=None):
        if values is None:
            n = 1
            for d in shape:
                n *= d
            seed = len(self.order)
            values = [(((i * 7 + seed * 13) % 13) - 6) / 8.0 for i in range(n)]
        assert len(values) == _count(shape), name
        self.order.append(name)
        self.shape[name] = tuple(shape)
        self.data[name] = list(values)
        return self

    def add_same(self, name, source):
        self.add(name, self.shape[source], self.data[source])

    def clone(self):
        t = Tensors()
        t.order = list(self.order)
        t.data = {k: list(v) for k, v in self.data.items()}
        t.shape = dict(self.shape)
        return t


def _count(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def rows(t, name):
    """The tensor as a list of rows along dim 0, each a flat list."""
    shape = t.shape[name]
    inner = _count(shape[1:])
    flat = t.data[name]
    return [flat[i * inner:(i + 1) * inner] for i in range(shape[0])]


def flatten(rs):
    return [x for r in rs for x in r]


def columns(t, name, start, count):
    """Columns [start, start+count) of a 2-D tensor, as a flat list in row-major order."""
    return flatten([r[start:start + count] for r in rows(t, name)])


# ----------------------------------------------------------------------------- safetensors

def write_safetensors(path, t, names=None):
    names = list(t.order if names is None else names)
    header = {}
    offset = 0
    blob = bytearray()
    for name in names:
        shape = t.shape[name]
        values = t.data[name]
        raw = struct.pack("<%df" % len(values), *values)
        header[name] = {"dtype": "F32", "shape": list(shape),
                        "data_offsets": [offset, offset + len(raw)]}
        blob += raw
        offset += len(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
        f.write(blob)


def write_hf(dirname, config, t):
    os.makedirs(dirname, exist_ok=True)
    with open(os.path.join(dirname, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(dirname, "tokenizer.json"), "w") as f:
        json.dump(TOKENIZER, f)
    with open(os.path.join(dirname, "tokenizer_config.json"), "w") as f:
        json.dump(TOKENIZER_CONFIG, f)
    write_safetensors(os.path.join(dirname, "model.safetensors"), t)


# ----------------------------------------------------------------------------- tensorizer

MAGIC = b"|TZR|"
META_AREA = 256 * 1024
TENSOR_TYPE_PARAM = 0


def tensorizer_header(name, shape, dtype, data_len, crypt_info=None):
    """One tensorizer v1 per-tensor header (32-bit dimensions, CRC32 and SHA256)."""
    name_b = name.encode("utf-8")
    dtype_b = dtype.encode("utf-8")
    has_crc = crypt_info is None
    hashes = b""
    if has_crc:
        hashes += struct.pack("<BBI", 0, 4, 0)          # CRC32 placeholder
    hashes += struct.pack("<BB32s", 1, 32, bytes(32))  # SHA256 placeholder
    variable = struct.pack("<%dsB%dsB%dI" % (len(name_b), len(dtype_b), len(shape)),
                           name_b, len(dtype_b), dtype_b, len(shape), *shape)
    crypt = b""
    if crypt_info is not None:
        crypt = struct.pack("<Q", len(crypt_info)) + crypt_info
    prefix = struct.pack("<QHBH", 0, 0, TENSOR_TYPE_PARAM, len(name_b))
    hash_header = struct.pack("<HB", len(hashes) + 1, 2 if has_crc else 1)
    trailer = struct.pack("<q", data_len)
    body = prefix + variable + hash_header + hashes + crypt + trailer
    header = struct.pack("<Q", len(body)) + body[8:]

    hashes_at = 13 + len(variable) + 3
    crc_at = None
    sha_at = hashes_at + 2
    if has_crc:
        crc_at = hashes_at + 2
        sha_at = hashes_at + 8
    return header, crc_at, sha_at


def write_tensorizer(path, t, names=None, key=None, chunk_size=None):
    """Writes a v1 tensorizer file, or a v3 encrypted one when key is given.

    The SHA256/CRC32 fields are filled the way TensorSerializer fills them: over the header
    with the hash fields zeroed, followed by the tensor data.
    """
    names = list(t.order if names is None else names)
    header_pos = 5 + 4 + 56 + 8 + META_AREA
    entries = []
    bodies = []
    encrypted = key is not None
    version = 3 if encrypted else 1
    pos = header_pos
    total = 0
    for name in names:
        shape = t.shape[name]
        raw = struct.pack("<%df" % len(t.data[name]), *t.data[name])
        dtype = "<f4"
        crypt_info = None
        if encrypted and chunk_size is not None and len(raw) > chunk_size:
            nonce = _nonce(name)
            macs = []
            data = b""
            current = nonce
            for at in range(0, len(raw), chunk_size):
                sealed = crypto_secretbox_easy(raw[at:at + chunk_size], current, key)
                macs.append(sealed[:16])
                data += sealed[16:]
                current = _increment(current)
            payload = (struct.pack("<HQ24sQ", 2, chunk_size, nonce, len(macs))
                       + b"".join(macs))
            crypt_info = struct.pack("<H", 1) + struct.pack("<Q", len(payload)) + payload
        elif encrypted:
            nonce = _nonce(name)
            sealed = crypto_secretbox_easy(raw, nonce, key)
            mac, data = sealed[:16], sealed[16:]
            payload = struct.pack("<H24s16s", 3, nonce, mac)
            crypt_info = struct.pack("<H", 1) + struct.pack("<Q", len(payload)) + payload
        else:
            data = raw
        header, crc_at, sha_at = tensorizer_header(
            name, shape, dtype, len(data), crypt_info)
        header = bytearray(header)
        zeros_and_data = bytes(header) + data
        if crc_at is not None:
            struct.pack_into("<I", header, crc_at, _crc32(zeros_and_data))
        struct.pack_into("<32s", header, sha_at, hashlib.sha256(zeros_and_data).digest())
        entries.append((name, shape, dtype, pos, pos + len(header), len(data)))
        bodies.append(bytes(header) + data)
        pos += len(header) + len(data)
        total += len(data)

    metadata = bytearray()
    for name, shape, dtype, header_offset, data_offset, data_length in entries:
        name_b = name.encode("utf-8")
        dtype_b = dtype.encode("utf-8")
        metadata += struct.pack("<H%dsBB%dsB%dI" % (len(name_b), len(dtype_b), len(shape)),
                                len(name_b), name_b, TENSOR_TYPE_PARAM, len(dtype_b), dtype_b,
                                len(shape), *shape)
        metadata += struct.pack("<QQQ", header_offset, data_offset, data_length)
    assert len(metadata) <= META_AREA

    flags = bytes(32)
    file_header = struct.pack("<32sQ8xQ", flags, total, len(entries))
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", version))
        f.write(file_header)
        f.write(struct.pack("<Q", META_AREA))
        f.write(bytes(metadata))
        f.write(bytes(META_AREA - len(metadata)))
        for body in bodies:
            f.write(body)


def _nonce(name):
    return hashlib.sha256(("nonce:" + name).encode("utf-8")).digest()[:24]


def _increment(nonce):
    return (int.from_bytes(nonce, "little") + 1).to_bytes(len(nonce), "little")


def _crc32_byte(crc, byte):
    crc ^= byte
    for _ in range(8):
        crc = (crc >> 1) ^ (0xEDB88320 if crc & 1 else 0)
    return crc


def _crc32(data):
    crc = 0xFFFFFFFF
    for b in data:
        crc = _crc32_byte(crc, b)
    return crc ^ 0xFFFFFFFF


# ----------------------------------------------------------------------------- Llama

LLAMA_DIM = 8
LLAMA_HEADS = 2
LLAMA_KV = 1
LLAMA_HEAD_DIM = LLAMA_DIM // LLAMA_HEADS
LLAMA_FFN = 2 * LLAMA_DIM
LLAMA_VOCAB = 8
LLAMA_PADDED = 10
LLAMA_LAYERS = 2
TP = 2

LLAMA_CONFIG = {
    "architectures": ["LlamaForCausalLM"], "model_type": "llama",
    "hidden_size": LLAMA_DIM, "intermediate_size": LLAMA_FFN,
    "num_hidden_layers": LLAMA_LAYERS, "num_attention_heads": LLAMA_HEADS,
    "num_key_value_heads": LLAMA_KV, "max_position_embeddings": 128,
    "rms_norm_eps": 1e-05, "rope_theta": 10000.0, "vocab_size": LLAMA_VOCAB,
    "torch_dtype": "float32", "bos_token_id": 0, "eos_token_id": 1,
    "tie_word_embeddings": True,
}


def llama_tensors(tied=True):
    t = Tensors()
    t.add("model.embed_tokens.weight", (LLAMA_VOCAB, LLAMA_DIM))
    for layer in range(LLAMA_LAYERS):
        p = "model.layers.%d." % layer
        t.add(p + "input_layernorm.weight", (LLAMA_DIM,))
        t.add(p + "self_attn.q_proj.weight", (LLAMA_DIM, LLAMA_DIM))
        t.add(p + "self_attn.k_proj.weight", (LLAMA_KV * LLAMA_HEAD_DIM, LLAMA_DIM))
        t.add(p + "self_attn.v_proj.weight", (LLAMA_KV * LLAMA_HEAD_DIM, LLAMA_DIM))
        t.add(p + "self_attn.o_proj.weight", (LLAMA_DIM, LLAMA_DIM))
        t.add(p + "post_attention_layernorm.weight", (LLAMA_DIM,))
        t.add(p + "mlp.gate_proj.weight", (LLAMA_FFN, LLAMA_DIM))
        t.add(p + "mlp.up_proj.weight", (LLAMA_FFN, LLAMA_DIM))
        t.add(p + "mlp.down_proj.weight", (LLAMA_DIM, LLAMA_FFN))
    t.add("model.norm.weight", (LLAMA_DIM,))
    if tied:
        t.add_same("lm_head.weight", "model.embed_tokens.weight")
    else:
        t.add("lm_head.weight", (LLAMA_VOCAB, LLAMA_DIM))
    return t


def padded(t, name, padded_rows):
    """The tensor's rows followed by zero rows up to the padded vocabulary."""
    rs = rows(t, name)
    inner = len(rs[0])
    rs = rs + [[0.0] * inner for _ in range(padded_rows - len(rs))]
    return flatten(rs)


def llama_vllm_rank(t, rank):
    """One tensor-parallel rank of Llama's vLLM module state dict (tied: no embed_tokens)."""
    out = Tensors()
    for layer in range(LLAMA_LAYERS):
        p = "model.layers.%d." % layer
        qkv_rows = []
        q = rows(t, p + "self_attn.q_proj.weight")
        k = rows(t, p + "self_attn.k_proj.weight")
        v = rows(t, p + "self_attn.v_proj.weight")
        heads = LLAMA_HEADS // TP
        qkv_rows += q[rank * heads * LLAMA_HEAD_DIM:(rank + 1) * heads * LLAMA_HEAD_DIM]
        qkv_rows += k          # one key/value head, replicated on every rank
        qkv_rows += v
        out.add(p + "self_attn.qkv_proj.weight", (len(qkv_rows), LLAMA_DIM), flatten(qkv_rows))
        out.add(p + "self_attn.o_proj.weight", (LLAMA_DIM, LLAMA_DIM // TP),
                columns(t, p + "self_attn.o_proj.weight", rank * (LLAMA_DIM // TP), LLAMA_DIM // TP))
        out.add(p + "input_layernorm.weight", (LLAMA_DIM,), t.data[p + "input_layernorm.weight"])
        out.add(p + "post_attention_layernorm.weight", (LLAMA_DIM,),
                t.data[p + "post_attention_layernorm.weight"])
        gate = rows(t, p + "mlp.gate_proj.weight")
        up = rows(t, p + "mlp.up_proj.weight")
        half = LLAMA_FFN // TP
        gu = gate[rank * half:(rank + 1) * half] + up[rank * half:(rank + 1) * half]
        out.add(p + "mlp.gate_up_proj.weight", (2 * half, LLAMA_DIM), flatten(gu))
        out.add(p + "mlp.down_proj.weight", (LLAMA_DIM, LLAMA_FFN // TP),
                columns(t, p + "mlp.down_proj.weight", rank * (LLAMA_FFN // TP), LLAMA_FFN // TP))
    out.add("model.norm.weight", (LLAMA_DIM,), t.data["model.norm.weight"])
    # vLLM ties lm_head to the token embedding and its state-dict filter keeps only `lm_head.weight`.
    out.add("lm_head.weight", (LLAMA_PADDED, LLAMA_DIM),
            padded(t, "lm_head.weight", LLAMA_PADDED))
    return out


# ----------------------------------------------------------------------------- Qwen3MoE

QWEN_DIM = 8
QWEN_HEADS = 2
QWEN_KV = 1
QWEN_HEAD_DIM = QWEN_DIM // QWEN_HEADS
QWEN_EXPERTS = 2
QWEN_EXPERT_FFN = 4
QWEN_VOCAB = 8
QWEN_PADDED = 10
QWEN_LAYERS = 2

QWEN_CONFIG = {
    "architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe",
    "hidden_size": QWEN_DIM, "intermediate_size": 16,
    "num_attention_heads": QWEN_HEADS, "num_key_value_heads": QWEN_KV,
    "num_hidden_layers": QWEN_LAYERS, "max_position_embeddings": 64,
    "rms_norm_eps": 1e-06, "rope_theta": 1000000.0, "vocab_size": QWEN_VOCAB,
    "num_experts": QWEN_EXPERTS, "moe_intermediate_size": QWEN_EXPERT_FFN,
    "num_experts_per_tok": 1, "torch_dtype": "float32",
    "tie_word_embeddings": False,
}


def qwen_tensors():
    t = Tensors()
    t.add("model.embed_tokens.weight", (QWEN_VOCAB, QWEN_DIM))
    for layer in range(QWEN_LAYERS):
        p = "model.layers.%d." % layer
        t.add(p + "input_layernorm.weight", (QWEN_DIM,))
        t.add(p + "self_attn.q_proj.weight", (QWEN_DIM, QWEN_DIM))
        t.add(p + "self_attn.k_proj.weight", (QWEN_KV * QWEN_HEAD_DIM, QWEN_DIM))
        t.add(p + "self_attn.v_proj.weight", (QWEN_KV * QWEN_HEAD_DIM, QWEN_DIM))
        t.add(p + "self_attn.o_proj.weight", (QWEN_DIM, QWEN_DIM))
        t.add(p + "self_attn.q_norm.weight", (QWEN_HEAD_DIM,))
        t.add(p + "self_attn.k_norm.weight", (QWEN_HEAD_DIM,))
        t.add(p + "post_attention_layernorm.weight", (QWEN_DIM,))
        t.add(p + "mlp.gate.weight", (QWEN_EXPERTS, QWEN_DIM))
        for x in range(QWEN_EXPERTS):
            t.add(p + "mlp.experts.%d.down_proj.weight" % x, (QWEN_DIM, QWEN_EXPERT_FFN))
            t.add(p + "mlp.experts.%d.gate_proj.weight" % x, (QWEN_EXPERT_FFN, QWEN_DIM))
            t.add(p + "mlp.experts.%d.up_proj.weight" % x, (QWEN_EXPERT_FFN, QWEN_DIM))
    t.add("model.norm.weight", (QWEN_DIM,))
    t.add("lm_head.weight", (QWEN_VOCAB, QWEN_DIM))
    return t


def expert_stack(t, prefix, part):
    """[experts, out, in] of one per-expert projection, in expert order."""
    return [rows(t, prefix + "mlp.experts.%d.%s.weight" % (x, part)) for x in range(QWEN_EXPERTS)]


def qwen_vllm_rank(t, rank):
    out = Tensors()
    for layer in range(QWEN_LAYERS):
        p = "model.layers.%d." % layer
        qkv = []
        q = rows(t, p + "self_attn.q_proj.weight")
        k = rows(t, p + "self_attn.k_proj.weight")
        v = rows(t, p + "self_attn.v_proj.weight")
        heads = QWEN_HEADS // TP
        qkv += q[rank * heads * QWEN_HEAD_DIM:(rank + 1) * heads * QWEN_HEAD_DIM]
        qkv += k
        qkv += v
        out.add(p + "self_attn.qkv_proj.weight", (len(qkv), QWEN_DIM), flatten(qkv))
        out.add(p + "self_attn.o_proj.weight", (QWEN_DIM, QWEN_DIM // TP),
                columns(t, p + "self_attn.o_proj.weight", rank * (QWEN_DIM // TP), QWEN_DIM // TP))
        out.add(p + "input_layernorm.weight", (QWEN_DIM,), t.data[p + "input_layernorm.weight"])
        out.add(p + "post_attention_layernorm.weight", (QWEN_DIM,),
                t.data[p + "post_attention_layernorm.weight"])
        out.add(p + "self_attn.q_norm.weight", (QWEN_HEAD_DIM,), t.data[p + "self_attn.q_norm.weight"])
        out.add(p + "self_attn.k_norm.weight", (QWEN_HEAD_DIM,), t.data[p + "self_attn.k_norm.weight"])
        out.add(p + "mlp.gate.weight", (QWEN_EXPERTS, QWEN_DIM), t.data[p + "mlp.gate.weight"])
        # FusedMoE w13: each rank holds its gate rows then its up rows per expert.
        w13 = []
        for x in range(QWEN_EXPERTS):
            gate = rows(t, p + "mlp.experts.%d.gate_proj.weight" % x)
            up = rows(t, p + "mlp.experts.%d.up_proj.weight" % x)
            half = QWEN_EXPERT_FFN // TP
            w13.append(gate[rank * half:(rank + 1) * half] + up[rank * half:(rank + 1) * half])
        out.add(p + "mlp.experts.w13_weight", (QWEN_EXPERTS, 2 * (QWEN_EXPERT_FFN // TP), QWEN_DIM),
                flatten([r for e in w13 for r in e]))
        w2 = []
        for x in range(QWEN_EXPERTS):
            w2.append(columns(t, p + "mlp.experts.%d.down_proj.weight" % x,
                              rank * (QWEN_EXPERT_FFN // TP), QWEN_EXPERT_FFN // TP))
        out.add(p + "mlp.experts.w2_weight", (QWEN_EXPERTS, QWEN_DIM, QWEN_EXPERT_FFN // TP),
                [x for e in w2 for x in e])
    out.add("model.norm.weight", (QWEN_DIM,), t.data["model.norm.weight"])
    out.add("model.embed_tokens.weight", (QWEN_PADDED, QWEN_DIM),
            padded(t, "model.embed_tokens.weight", QWEN_PADDED))
    out.add("lm_head.weight", (QWEN_PADDED, QWEN_DIM), padded(t, "lm_head.weight", QWEN_PADDED))
    return out


# ----------------------------------------------------------------------------- output

KEY = bytes(range(32))


def write_sharded(dirname, config, t):
    os.makedirs(dirname, exist_ok=True)
    with open(os.path.join(dirname, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(dirname, "tokenizer.json"), "w") as f:
        json.dump(TOKENIZER, f)
    with open(os.path.join(dirname, "tokenizer_config.json"), "w") as f:
        json.dump(TOKENIZER_CONFIG, f)
    ranks = [llama_vllm_rank(t, r) for r in range(TP)] if config is LLAMA_CONFIG \
        else [qwen_vllm_rank(t, r) for r in range(TP)]
    for r, rank in enumerate(ranks):
        write_safetensors(os.path.join(dirname, "model-rank-%d-part-0.safetensors" % r), rank)


def write_tensorizer_dir(dirname, config, ranks):
    os.makedirs(dirname, exist_ok=True)
    with open(os.path.join(dirname, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(dirname, "tokenizer.json"), "w") as f:
        json.dump(TOKENIZER, f)
    with open(os.path.join(dirname, "tokenizer_config.json"), "w") as f:
        json.dump(TOKENIZER_CONFIG, f)
    if len(ranks) == 1:
        write_tensorizer(os.path.join(dirname, "model.tensors"), ranks[0])
    else:
        for r, rank in enumerate(ranks):
            write_tensorizer(os.path.join(dirname, "model-%04d.tensors" % r), rank)


def main():
    os.makedirs(OUT, exist_ok=True)

    llama = llama_tensors(tied=True)
    llama_hf = os.path.join(OUT, "llama-tp2", "hf")
    write_hf(llama_hf, LLAMA_CONFIG, llama)
    write_sharded(os.path.join(OUT, "llama-tp2", "vllm"), LLAMA_CONFIG, llama)

    qwen = qwen_tensors()
    write_hf(os.path.join(OUT, "qwen3moe-tp2", "hf"), QWEN_CONFIG, qwen)
    write_sharded(os.path.join(OUT, "qwen3moe-tp2", "vllm"), QWEN_CONFIG, qwen)
    write_tensorizer_dir(os.path.join(OUT, "qwen3moe-tp2", "tensorizer"), QWEN_CONFIG,
                         [qwen_vllm_rank(qwen, r) for r in range(TP)])

    untied = llama_tensors(tied=False)
    write_hf(os.path.join(OUT, "llama-untied", "hf"), dict(LLAMA_CONFIG, tie_word_embeddings=False),
             untied)
    write_tensorizer_dir(os.path.join(OUT, "llama-untied", "tensorizer"),
                         dict(LLAMA_CONFIG, tie_word_embeddings=False), [untied])
    enc = os.path.join(OUT, "llama-encrypted", "tensorizer")
    os.makedirs(enc, exist_ok=True)
    with open(os.path.join(enc, "config.json"), "w") as f:
        json.dump(dict(LLAMA_CONFIG, tie_word_embeddings=False), f, indent=2)
    with open(os.path.join(enc, "tokenizer.json"), "w") as f:
        json.dump(TOKENIZER, f)
    with open(os.path.join(enc, "tokenizer_config.json"), "w") as f:
        json.dump(TOKENIZER_CONFIG, f)
    write_tensorizer(os.path.join(enc, "model.tensors"), untied, key=KEY)

    chunked = Tensors()
    chunked.add("wide.weight", (64, 8), [i / 1000.0 for i in range(64 * 8)])
    chunk_dir = os.path.join(OUT, "tensorizer-chunked")
    os.makedirs(chunk_dir, exist_ok=True)
    write_tensorizer(os.path.join(chunk_dir, "model.tensors"), chunked, key=KEY, chunk_size=1024)

    print("fixtures written to", OUT)


if __name__ == "__main__":
    sys.exit(main())
