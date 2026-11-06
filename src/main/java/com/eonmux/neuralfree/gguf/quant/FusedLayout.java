package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.quant.FusedProjections.Layout;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collections;
import java.util.List;
import java.util.function.LongUnaryOperator;

/**
 * The virtual tensor vLLM quantizes for the members of one fused projection ({@link FusedProjections}), or for one
 * tensor alone: its row length, expert count and rows, and where each member's rows lie in it. Expert-major: expert
 * {@code e} keeps virtual rows {@code [e * rows / experts, (e + 1) * rows / experts)}.
 * <ul>
 *   <li>{@link Layout#ROWS}: the members' rows in fusion order. A linear member whose rows the converter reordered
 *       ({@link RowTransforms#storedSource}, the rope permutation of LLaMA-family q/k and the GDN regrouping)
 *       sits at the rows of its checkpoint tensor, in the order vLLM loads them; members read from one checkpoint
 *       tensor (Qwen3-Next's {@code in_proj_qkvz}) share its rows. A stacked routed-experts tensor keeps its experts.</li>
 *   <li>{@link Layout#EXPERT_SHARDS}: per expert, each member's rows starting on a multiple of 128 rows, the zero
 *       padding vLLM gives each shard of a routed-experts weight.</li>
 *   <li>{@link Layout#MLA_KV_B}: {@code kv_b [heads * (nope + v), kv_lora]}, {@code attn_k_b}'s rows running down
 *       its columns ({@link Placement#transposed()}) and {@code attn_v_b}'s rows its value rows.</li>
 * </ul>
 * Immutable.
 */
final class FusedLayout {

    private static final Logger LOG = LogManager.getLogger(FusedLayout.class);

    /** The rows a routed-expert shard is padded to in a fused group (vLLM rounds the intermediate size up to 128). */
    static final int EXPERT_SHARD_ROWS = 128;

    final int inner;
    final long experts;
    final long rows;
    final List<Placement> placements;

    private FusedLayout(int inner, long experts, long rows, List<Placement> placements) {
        if (rows > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(rows + " fused rows do not fit one int-sized quantization scan");
        }
        this.inner = inner;
        this.experts = experts;
        this.rows = rows;
        this.placements = Collections.unmodifiableList(new ArrayList<>(placements));
    }

    /**
     * Where the rows of one member lie in the virtual tensor: a row is virtual row {@link #row}, or, for a
     * transposed member, the run of column {@link #column} down from virtual row {@link #row}, one value per
     * virtual row. Immutable.
     */
    static final class Placement {
        private final LongUnaryOperator row;
        private final LongUnaryOperator column;
        private final long columnAlign;

        private Placement(LongUnaryOperator row, LongUnaryOperator column, long columnAlign) {
            this.row = row;
            this.column = column;
            this.columnAlign = columnAlign;
        }

        static Placement rows(LongUnaryOperator row) {
            return new Placement(row, null, 0);
        }

        /** A transposed member whose runs start at virtual rows that are all multiples of {@code align}. */
        static Placement columns(LongUnaryOperator start, LongUnaryOperator column, long align) {
            return new Placement(start, column, align);
        }

        /** The virtual row of member row {@code r}, or the first virtual row of its run when transposed. */
        long row(long r) {
            return row.applyAsLong(r);
        }

        /** Whether the member's rows run down the virtual tensor's columns. */
        boolean transposed() {
            return column != null;
        }

        /** The virtual column of member row {@code r} of a transposed member. */
        int column(long r) {
            return (int) column.applyAsLong(r);
        }

        /** A transposed member's run starts are multiples of this (0 when they all start at row 0). */
        long columnAlign() {
            return columnAlign;
        }
    }

    /**
     * The layout of {@code srcs} named {@code names} (fusion order) as {@code layout}; {@code moe} for a routed
     * experts layer, whose 3-D members stack their experts on dim 0.
     *
     * @throws IllegalArgumentException when the members differ in row length or expert count, or do not form the
     *                                  layout's shapes
     */
    static FusedLayout of(List<String> names, List<RowSource> srcs, Layout layout, boolean moe) {
        switch (layout) {
            case MLA_KV_B:
                return mlaKvB(names, srcs);
            case EXPERT_SHARDS:
                return concatenated(names, srcs, moe, EXPERT_SHARD_ROWS);
            default:
                if (moe) {
                    return concatenated(names, srcs, true, 1);
                }
                FusedLayout traced = traced(names, srcs);
                return traced != null ? traced : concatenated(names, srcs, false, 1);
        }
    }

    /** The members' rows concatenated per expert, each member starting on a multiple of {@code align} rows. */
    private static FusedLayout concatenated(List<String> names, List<RowSource> srcs, boolean moe, int align) {
        int inner = -1;
        long experts = -1;
        long offset = 0;
        List<long[]> spans = new ArrayList<>();
        for (int m = 0; m < srcs.size(); m++) {
            long[] shape = srcs.get(m).shape();
            int in = innerOf(names.get(m), shape);
            long ex = moe && shape.length == 3 ? shape[0] : 1;
            long rows = srcs.get(m).rows();
            if ((inner >= 0 && (in != inner || ex != experts)) || ex <= 0 || rows % ex != 0) {
                throw new IllegalArgumentException("the fused group " + names + " mixes row lengths or expert"
                        + " counts: " + names.get(m) + " is " + Arrays.toString(shape));
            }
            inner = in;
            experts = ex;
            spans.add(new long[]{rows / ex, offset});
            offset = (offset + rows / ex + align - 1) / align * align;
        }
        long perExpert = offset;
        List<Placement> out = new ArrayList<>();
        for (long[] span : spans) {
            long n = span[0];
            long at = span[1];
            out.add(Placement.rows(r -> r / n * perExpert + at + r % n));
        }
        return new FusedLayout(inner, experts, Math.multiplyExact(experts, perExpert), out);
    }

    /**
     * The linear members at the rows of their checkpoint tensors ({@link RowTransforms#storedSource}), the
     * checkpoint tensors concatenated in the order the members first read them; null when the members' rows are
     * no exact permutation of those tensors' rows, or differ in length.
     */
    private static FusedLayout traced(List<String> names, List<RowSource> srcs) {
        int inner = innerOf(names.get(0), srcs.get(0).shape());
        List<RowSource> bases = new ArrayList<>();
        List<Long> offsets = new ArrayList<>();
        long total = 0;
        int[] baseOf = new int[srcs.size()];
        for (int m = 0; m < srcs.size(); m++) {
            if (innerOf(names.get(m), srcs.get(m).shape()) != inner) {
                throw new IllegalArgumentException("the fused group " + names + " mixes row lengths: "
                        + names.get(m) + " is " + Arrays.toString(srcs.get(m).shape()));
            }
            RowSource base = RowTransforms.storedSource(srcs.get(m));
            int b = indexOfIdentity(bases, base);
            if (b < 0) {
                if (innerOf(names.get(m), base.shape()) != inner) {
                    return untraced(names, "its checkpoint tensor " + base.describe() + " has rows of another length");
                }
                b = bases.size();
                bases.add(base);
                offsets.add(total);
                total += base.rows();
            }
            baseOf[m] = b;
        }
        if (total > Integer.MAX_VALUE) {
            return untraced(names, "their checkpoint tensors hold " + total + " rows");
        }
        BitSet seen = new BitSet((int) total);
        List<Placement> out = new ArrayList<>();
        for (int m = 0; m < srcs.size(); m++) {
            RowSource src = srcs.get(m);
            long at = offsets.get(baseOf[m]);
            long limit = bases.get(baseOf[m]).rows();
            for (long r = 0; r < src.rows(); r++) {
                long s = RowTransforms.storedRow(src, r);
                if (s < 0 || s >= limit || seen.get((int) (at + s))) {
                    return untraced(names, names.get(m) + " row " + r + " reads checkpoint row " + s
                            + " out of range or twice");
                }
                seen.set((int) (at + s));
            }
            out.add(Placement.rows(r -> at + RowTransforms.storedRow(src, r)));
        }
        if (seen.cardinality() != total) {
            return untraced(names, "the members read " + seen.cardinality() + " of their checkpoint tensors' "
                    + total + " rows");
        }
        return new FusedLayout(inner, 1, total, out);
    }

    private static FusedLayout untraced(List<String> names, String why) {
        LOG.warn("online quantization of {}: the scales follow the converted row order, not the checkpoint's: {}",
                names, why);
        return null;
    }

    /**
     * {@code kv_b}: {@code attn_k_b [heads, kv_lora, nope]} row {@code (h, c)} is column {@code c} of virtual rows
     * {@code [h * (nope + v), h * (nope + v) + nope)}, and {@code attn_v_b [heads, v, kv_lora]} row {@code (h, i)}
     * is virtual row {@code h * (nope + v) + nope + i}.
     */
    private static FusedLayout mlaKvB(List<String> names, List<RowSource> srcs) {
        if (srcs.size() != 2 || !FusedProjections.transposed(names.get(0))
                || FusedProjections.transposed(names.get(1))) {
            throw new IllegalArgumentException("kv_b fuses attn_k_b and attn_v_b in that order, got " + names);
        }
        long[] k = srcs.get(0).shape();
        long[] v = srcs.get(1).shape();
        if (k.length != 3 || v.length != 3 || k[0] != v[0] || k[1] != v[2] || k[0] <= 0) {
            throw new IllegalArgumentException("the kv_b group " + names + " is not attn_k_b [heads, kv_lora, nope]"
                    + " and attn_v_b [heads, v, kv_lora]: " + Arrays.toString(k) + ", " + Arrays.toString(v));
        }
        long heads = k[0];
        long kvLora = k[1];
        long nope = k[2];
        long vHead = v[1];
        long stride = nope + vHead;
        int inner = innerOf(names.get(1), v);
        List<Placement> out = new ArrayList<>();
        out.add(Placement.columns(r -> r / kvLora * stride, r -> r % kvLora, heads > 1 ? stride : 0));
        out.add(Placement.rows(r -> r / vHead * stride + nope + r % vHead));
        return new FusedLayout(inner, 1, Math.multiplyExact(heads, stride), out);
    }

    private static int indexOfIdentity(List<RowSource> list, RowSource s) {
        for (int i = 0; i < list.size(); i++) {
            if (list.get(i) == s) {
                return i;
            }
        }
        return -1;
    }

    /** The row length of a tensor of {@code shape}, as an int. */
    static int innerOf(String ggufName, long[] shape) {
        long inner = shape.length == 0 ? 1 : shape[shape.length - 1];
        if (inner > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(ggufName + ": a row of " + inner
                    + " values does not fit one int-sized requantization row");
        }
        return (int) inner;
    }
}
