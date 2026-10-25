package com.eonmux.neuralfree.gguf.hf.tensor;

import java.util.Arrays;
import java.util.function.LongUnaryOperator;

/**
 * Fused attention projections: the layouts checkpoints fuse Q, K and V (and Qwen3-Next's Z gate) into one tensor, and
 * DeepSeek's MLA {@code kv_b_proj} split. Every split and regroup is a move of dim 0, so it is exact on every
 * format and row-local; a 1-D bias moves element by element.
 */
public final class QkvLayouts {

    /** How a checkpoint orders the rows of a fused QKV tensor. */
    public enum Layout {
        /** {@code [Q; K; V]} in three equal parts ({@code torch.chunk(w, 3)}). */
        THIRDS,
        /** {@code [Q (nHead*hd); K (nKv*hd); V (nKv*hd)]}. */
        SIZES,
        /** Bloom / GPT-NeoX {@code [nHead, 3, hd, ...]} (LC/conversion/bloom.py:42-66). */
        PER_HEAD_INTERLEAVED,
        /** Falcon {@code [nKv, nHead/nKv + 2, hd, ...]}: each KV group's query heads, then its K and V heads. */
        GROUPED
    }

    private QkvLayouts() {
    }

    /**
     * The Q, K and V parts of a fused tensor, each with its heads in order.
     *
     * @throws IllegalArgumentException when dim 0 does not hold the heads the layout describes
     */
    public static RowSource[] splitQkv(RowSource src, Layout layout, int nHead, int nKv, int headDim) {
        long d0 = RowTransforms.dim0(src);
        if (layout == Layout.THIRDS) {
            if (d0 % 3 != 0) {
                throw new IllegalArgumentException(src.describe() + ": dim 0 (" + d0 + ") is not three equal parts");
            }
            return RowTransforms.splitRows(src, d0 / 3, d0 / 3, d0 / 3);
        }
        if (nHead <= 0 || nKv <= 0 || headDim <= 0 || nHead % nKv != 0) {
            throw new IllegalArgumentException(src.describe() + ": " + nHead + " query heads do not group over "
                    + nKv + " KV heads of width " + headDim);
        }
        long q = (long) nHead * headDim;
        long kv = (long) nKv * headDim;
        if (d0 != q + 2 * kv) {
            throw new IllegalArgumentException(src.describe() + ": dim 0 (" + d0 + ") is not " + nHead + " + 2 * "
                    + nKv + " heads of " + headDim);
        }
        switch (layout) {
            case SIZES:
                return RowTransforms.splitRows(src, q, kv, kv);
            case PER_HEAD_INTERLEAVED: {
                if (nKv != nHead) {
                    throw new IllegalArgumentException(src.describe() + ": the per-head interleaved layout needs as"
                            + " many KV heads as query heads, got " + nKv + " and " + nHead);
                }
                return new RowSource[]{
                        part(src, "q", q, e -> (e / headDim * 3) * headDim + e % headDim),
                        part(src, "k", kv, e -> (e / headDim * 3 + 1) * headDim + e % headDim),
                        part(src, "v", kv, e -> (e / headDim * 3 + 2) * headDim + e % headDim)};
            }
            case GROUPED: {
                int g = nHead / nKv;
                return new RowSource[]{
                        part(src, "q", q, e -> {
                            long h = e / headDim;
                            return ((h / g) * (g + 2) + h % g) * headDim + e % headDim;
                        }),
                        part(src, "k", kv, e -> ((e / headDim) * (g + 2) + g) * headDim + e % headDim),
                        part(src, "v", kv, e -> ((e / headDim) * (g + 2) + g + 1) * headDim + e % headDim)};
            }
            default:
                throw new IllegalStateException("unhandled layout " + layout);
        }
    }

    /**
     * {@code torch.cat(splitQkv(...))}: the fused tensor in {@code [Q; K; V]} order, as the GGUF {@code attn_qkv}
     * expects. THIRDS and SIZES are already in that order and are returned unchanged.
     */
    public static RowSource regroupQkv(RowSource src, Layout layout, int nHead, int nKv, int headDim) {
        if (layout == Layout.THIRDS || layout == Layout.SIZES) {
            splitQkv(src, layout, nHead, nKv, headDim);
            return src;
        }
        return RowTransforms.concatRows(Arrays.asList(splitQkv(src, layout, nHead, nKv, headDim)));
    }

    /**
     * Qwen3-Next {@code in_proj_qkvz} (LC/conversion/qwen.py:414-442): rows grouped per key head as
     * {@code [q (hk), k (hk), v (g*hv), z (g*hv)]} with {@code g = nVHeads / nKHeads}, regrouped to the fused
     * {@code attn_qkv} rows {@code [all q; all k; all v]} and the {@code attn_gate} rows (all z).
     *
     * @return {@code {qkv, z}}
     * @throws IllegalArgumentException when dim 0 does not hold the heads
     */
    public static RowSource[] splitQkvz(RowSource src, int nKHeads, int nVHeads, int headKDim, int headVDim) {
        if (nKHeads <= 0 || nVHeads <= 0 || headKDim <= 0 || headVDim <= 0 || nVHeads % nKHeads != 0) {
            throw new IllegalArgumentException(src.describe() + ": " + nVHeads + " value heads do not group over "
                    + nKHeads + " key heads");
        }
        long hk = headKDim;
        long gv = (long) (nVHeads / nKHeads) * headVDim;
        long w = 2 * hk + 2 * gv;
        long d0 = RowTransforms.dim0(src);
        if (d0 != nKHeads * w) {
            throw new IllegalArgumentException(src.describe() + ": dim 0 (" + d0 + ") is not " + nKHeads
                    + " key-head groups of " + w);
        }
        long qAll = nKHeads * hk;
        RowSource qkv = part(src, "qkv", 2 * qAll + nKHeads * gv, e -> {
            if (e < qAll) {
                return e / hk * w + e % hk;
            }
            if (e < 2 * qAll) {
                long f = e - qAll;
                return f / hk * w + hk + f % hk;
            }
            long f = e - 2 * qAll;
            return f / gv * w + 2 * hk + f % gv;
        });
        RowSource z = part(src, "z", nKHeads * gv, e -> e / gv * w + 2 * hk + gv + e % gv);
        return new RowSource[]{qkv, z};
    }

    /**
     * DeepSeek MLA {@code kv_b_proj} (LC/conversion/deepseek.py:434-450):
     * {@code w.view(nHead, qkNope + vHead, kvLora)} split into {@code k_b = w[:, :qkNope].transpose(1, 2)}
     * ({@code [nHead, kvLora, qkNope]}) and {@code v_b = w[:, qkNope:]} ({@code [nHead, vHead, kvLora]}). The
     * transposed {@code k_b} is not row-local.
     *
     * @return {@code {k_b, v_b}}
     * @throws IllegalArgumentException when the shape is not {@code [nHead * (qkNope + vHead), kvLora]}
     */
    public static RowSource[] mlaKvB(RowSource src, int nHead, int qkNope, int vHead, int kvLora) {
        long[] shape = src.shape();
        if (shape.length != 2 || nHead <= 0 || qkNope <= 0 || vHead <= 0
                || shape[0] != (long) nHead * (qkNope + vHead) || shape[1] != kvLora) {
            throw new IllegalArgumentException(src.describe() + ": kv_b " + Arrays.toString(shape) + " is not ["
                    + nHead + " * (" + qkNope + " + " + vHead + "), " + kvLora + "]");
        }
        RowSource heads = ElementTransforms.reshape(src, nHead, qkNope + vHead, kvLora);
        RowSource kb = ExpertTransforms.transposeLast2(RowTransforms.sliceDim(heads, 1, 0, qkNope));
        RowSource vb = RowTransforms.sliceDim(heads, 1, qkNope, vHead);
        return new RowSource[]{kb, vb};
    }

    private static RowSource part(RowSource src, String name, long n, LongUnaryOperator srcEntry) {
        return RowTransforms.mapDim0(src, name + " part", n, srcEntry);
    }
}
