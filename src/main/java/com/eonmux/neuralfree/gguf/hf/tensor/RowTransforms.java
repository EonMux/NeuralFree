package com.eonmux.neuralfree.gguf.hf.tensor;

import com.eonmux.neuralfree.gguf.quant.ActQuantScheme;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.function.LongUnaryOperator;

/**
 * Moves along the outermost dimension (PyTorch dim 0) of a tensor: the converter's Q/K permutes, fused-tensor
 * slices and splits, concatenations and padding or truncation.
 * "Rows" in these factories are the entries of dim 0: the rows of a 2-D weight, the experts of a 3-D stack, the
 * elements of a 1-D bias. Whole entries are copied byte for byte, so every op is exact on every format, block types
 * included; a 1-D source is moved element by element through {@link ColumnTransforms}. Every producer is row-local.
 *
 * <p>The converter's reorders of one stored tensor's rows that vLLM does not make (the Q/K rope permutes, the
 * regrouping of fused GDN projections and their value heads) keep their {@link #storedSource stored rows}
 * traceable, so online quantization can tile scales in vLLM's (stored) row order.</p>
 *
 * <p>Sibling values of quantized sources follow the move: a per-tensor scale is kept, per-expert scales of a stack are
 * re-indexed with its experts, and the input-side values (input scale, zero point, act-order permutation, activation
 * scheme) are kept because the input dimension is unchanged. A low-rank factor pair ({@link LoraFactorSource}) is
 * moved through {@link LoraFactorOps}.</p>
 */
public final class RowTransforms {

    /** Output bytes an element producer handles per read of its inputs. */
    static final int CHUNK_BYTES = 4 << 20;

    private RowTransforms() {
    }

    // ------------------------------------------------------------------ Q/K row permutations

    /**
     * llama.cpp {@code LlamaModel.permute} (LC/conversion/llama.py:176-182): HF's rotate-half Q/K layout to ggml's
     * interleaved pairs, {@code w.reshape(n, 2, d0 / n / 2, ...).swapaxes(1, 2)} with {@code n = nHeadKv} when it
     * differs from {@code nHead}. Output entry {@code h*2*half + i*2 + j} is input entry
     * {@code h*2*half + j*half + i}. A 1-D bias is permuted element by element.
     *
     * @throws IllegalArgumentException when dim 0 is not a multiple of {@code 2 * n}
     */
    public static RowSource permuteQk(RowSource src, int nHead, int nHeadKv) {
        int n = nHeadKv > 0 && nHeadKv != nHead ? nHeadKv : nHead;
        return pairSwap(src, n, "permuteQk(" + nHead + ", " + nHeadKv + ")");
    }

    /**
     * Baichuan / InternLM {@code _reverse_hf_permute} (LC/conversion/baichuan.py:42-50): the same pair swap with
     * {@code n = nHead / nHeadKv} heads when the two counts differ (the converter's {@code n_head //= n_kv_head}).
     */
    public static RowSource reversePermute(RowSource src, int nHead, int nHeadKv) {
        if (nHead <= 0 || nHeadKv < 0) {
            throw new IllegalArgumentException("reversePermute needs positive head counts, got " + nHead + "/"
                    + nHeadKv);
        }
        int n = nHeadKv > 0 && nHeadKv != nHead ? nHead / nHeadKv : nHead;
        return pairSwap(src, n, "reversePermute(" + nHead + ", " + nHeadKv + ")");
    }

    /**
     * Inverse of transformers' {@code _permute_for_rope} (MuseGlimmer / MuseSpark {@code _unpermute_for_rope},
     * LC/conversion/muse_glimmer.py:14-23): the {@link #permuteQk} pair swap over {@code nHead} heads.
     */
    public static RowSource rotateHalfUnpermute(RowSource src, int nHead) {
        return pairSwap(src, nHead, "rotateHalfUnpermute(" + nHead + ")");
    }

    /**
     * GLM-4 {@code normal_to_neox} (LC/conversion/glm.py:54-73): inside each head of {@code headDim} entries, the
     * first {@code rotDim} entries go from interleaved pairs to halves (even entries, then odd ones) and the rest stay.
     *
     * @param nHead the head count dim 0 must hold ({@code d0 == nHead * headDim})
     * @throws IllegalArgumentException when {@code rotDim} is odd or larger than {@code headDim}, or dim 0 does not
     *                                  hold {@code nHead} heads
     */
    public static RowSource normalToNeoxPartial(RowSource src, int nHead, int headDim, int rotDim) {
        long d0 = dim0(src);
        if (nHead <= 0 || headDim <= 0 || rotDim < 0 || rotDim > headDim || rotDim % 2 != 0
                || d0 != (long) nHead * headDim) {
            throw new IllegalArgumentException(src.describe() + ": normalToNeoxPartial needs " + nHead + " heads of "
                    + headDim + " entries with an even rotary width " + rotDim + " <= the head width; dim 0 is " + d0);
        }
        int half = rotDim / 2;
        return mapDim0(src, "normalToNeoxPartial(" + nHead + ", " + headDim + ", " + rotDim + ")", d0, e -> {
            long h = e / headDim;
            int k = (int) (e % headDim);
            int in = k < half ? 2 * k : k < rotDim ? 2 * (k - half) + 1 : k;
            return h * headDim + in;
        });
    }

    /**
     * Talkie's inverse-RoPE sign absorption (LC/conversion/talkie.py:36-46): within each head of {@code headDim}
     * entries, the second half is negated. Applied as {@link ElementTransforms#mulRowsBy} with factors of +-1, so
     * scaled block formats stay exact.
     */
    public static RowSource negateSecondHalfRows(RowSource src, int headDim) {
        if (src instanceof LoraFactorSource) {
            return LoraFactorOps.onB((LoraFactorSource) src, b -> negateSecondHalfRows(b, headDim));
        }
        long[] shape = src.shape();
        if (shape.length != 2 || headDim <= 1 || headDim % 2 != 0 || shape[0] % headDim != 0) {
            throw new IllegalArgumentException(src.describe() + ": negateSecondHalfRows needs a 2-D tensor whose rows"
                    + " are whole heads of an even width " + headDim);
        }
        float[] f = new float[Math.toIntExact(shape[0])];
        for (int r = 0; r < f.length; r++) {
            f[r] = r % headDim < headDim / 2 ? 1f : -1f;
        }
        return ElementTransforms.mulRowsBy(src, f);
    }

    /** The pair swap behind {@link #permuteQk}: {@code reshape(n, 2, half, ...).swapaxes(1, 2)}. */
    private static RowSource pairSwap(RowSource src, int n, String what) {
        long d0 = dim0(src);
        if (n <= 0 || d0 % (2L * n) != 0) {
            throw new IllegalArgumentException(src.describe() + ": " + what + " needs dim 0 (" + d0
                    + ") to be a multiple of 2 * " + n);
        }
        long headWidth = d0 / n;
        long half = headWidth / 2;
        return reorderDim0(src, what, d0, e -> {
            long h = e / headWidth;
            long w = e % headWidth;
            return h * headWidth + (w % 2) * half + w / 2;
        });
    }

    /**
     * Dim 0 with the {@code order.length} units of {@code unit} entries from entry {@code from} on reordered:
     * output unit {@code u} is input unit {@code order[u]}; the entries before {@code from} stay. The stored rows
     * stay traceable ({@link #storedSource}).
     *
     * @throws IllegalArgumentException when the units do not fill dim 0 from {@code from} on, or {@code order} is
     *                                  not a permutation
     */
    public static RowSource permuteRowGroups(RowSource src, long from, int unit, int[] order) {
        long d0 = dim0(src);
        int[] idx = order.clone();
        boolean[] seen = new boolean[idx.length];
        for (int i : idx) {
            if (i < 0 || i >= idx.length || seen[i]) {
                throw new IllegalArgumentException(src.describe() + ": " + Arrays.toString(order)
                        + " is not a permutation of " + idx.length + " units");
            }
            seen[i] = true;
        }
        if (unit <= 0 || from < 0 || d0 - from != (long) idx.length * unit) {
            throw new IllegalArgumentException(src.describe() + ": dim 0 (" + d0 + ") is not " + from + " entries and "
                    + idx.length + " units of " + unit);
        }
        return reorderDim0(src, "permuteRowGroups(" + from + ", " + unit + ")", d0, e -> e < from ? e
                : from + (long) idx[(int) ((e - from) / unit)] * unit + (e - from) % unit);
    }

    /**
     * The tensor whose rows {@code src} reads through the converter's traceable row reorders
     * ({@link #reorderDim0}): the source {@link #storedRow} indexes. A source with no such reorder is its own.
     */
    public static RowSource storedSource(RowSource src) {
        RowSource s = Objects.requireNonNull(src, "src");
        while (s instanceof RowMap && ((RowMap) s).stored) {
            s = ((RowMap) s).parts.get(0);
        }
        return s;
    }

    /** The row of {@link #storedSource} that row {@code row} of {@code src} reads. */
    public static long storedRow(RowSource src, long row) {
        RowSource s = Objects.requireNonNull(src, "src");
        long r = row;
        while (s instanceof RowMap && ((RowMap) s).stored) {
            r = ((RowMap) s).index.row(r);
            s = ((RowMap) s).parts.get(0);
        }
        return r;
    }

    // ------------------------------------------------------------------ slices and splits

    /** {@code src[from : from + count]}; a slice of a tensor stored unchanged stays mapped in place. */
    public static RowSource sliceRows(RowSource src, long from, long count) {
        return sliceDim(src, 0, from, count);
    }

    /**
     * {@code torch.chunk(src, parts, dim=0)[index]}: chunks of {@code ceil(d0 / parts)} entries, the last one
     * shorter.
     *
     * @throws IllegalArgumentException when chunk {@code index} does not exist
     */
    public static RowSource chunkRows(RowSource src, int parts, int index) {
        long d0 = dim0(src);
        if (parts <= 0 || index < 0 || index >= parts) {
            throw new IllegalArgumentException(src.describe() + ": chunk " + index + " of " + parts
                    + " does not exist");
        }
        long size = (d0 + parts - 1) / parts;
        long from = size * index;
        if (from >= d0 && d0 > 0) {
            throw new IllegalArgumentException(src.describe() + ": torch.chunk of " + d0 + " entries into " + parts
                    + " has no chunk " + index);
        }
        return sliceRows(src, from, Math.min(size, d0 - from));
    }

    /**
     * {@code torch.split(src, sizes, dim=0)}.
     *
     * @throws IllegalArgumentException when the sizes do not add up to dim 0
     */
    public static RowSource[] splitRows(RowSource src, long... sizes) {
        long d0 = dim0(src);
        long sum = 0;
        for (long s : sizes) {
            if (s < 0) {
                throw new IllegalArgumentException(src.describe() + ": negative split size in "
                        + Arrays.toString(sizes));
            }
            sum = Math.addExact(sum, s);
        }
        if (sum != d0) {
            throw new IllegalArgumentException(src.describe() + ": split sizes " + Arrays.toString(sizes)
                    + " add up to " + sum + ", not dim 0 (" + d0 + ")");
        }
        RowSource[] out = new RowSource[sizes.length];
        long from = 0;
        for (int i = 0; i < sizes.length; i++) {
            out[i] = sliceRows(src, from, sizes[i]);
            from += sizes[i];
        }
        return out;
    }

    /** {@code src[offset::stride]} over dim 0. */
    public static RowSource interleaveRows(RowSource src, int stride, int offset) {
        long d0 = dim0(src);
        if (stride <= 0 || offset < 0 || offset >= stride) {
            throw new IllegalArgumentException(src.describe() + ": [" + offset + "::" + stride + "] is not a valid"
                    + " stride slice");
        }
        long n = d0 > offset ? (d0 - offset + stride - 1) / stride : 0;
        return mapDim0(src, "[" + offset + "::" + stride + "]", n, e -> offset + e * stride);
    }

    /**
     * {@code src[offset::stride]} over the flat {@code [rows, innermost]} rows of a rank-N source: only rows
     * regroup, so a packed stack keeps its block byte accounting.
     */
    public static RowSource interleaveFlatRows(RowSource src, int stride, int offset) {
        long d0 = src.rows();
        if (stride <= 0 || offset < 0 || offset >= stride) {
            throw new IllegalArgumentException(src.describe() + ": [" + offset + "::" + stride + "] is not a valid"
                    + " stride slice");
        }
        long n = d0 > offset ? (d0 - offset + stride - 1) / stride : 0;
        long[] outShape = {n, src.shape()[src.shape().length - 1]};
        RowIndex index = r -> offset + r * stride;
        return new RowMap(src.describe() + " [" + offset + "::" + stride + "]",
                Collections.singletonList(src), outShape, index, null, null);
    }

    /**
     * {@code src.narrow(dim, from, count)}: a slice along any dimension. The innermost dimension is a column slice
     * ({@link ColumnTransforms#sliceCols}); an outer one keeps whole rows.
     *
     * @param dim the dimension, negative values counting from the innermost ({@code -1})
     */
    public static RowSource sliceDim(RowSource src, int dim, long from, long count) {
        long[] shape = src.shape();
        int d = normalizeDim(src, dim);
        if (from < 0 || count < 0 || from > shape[d] - count) {
            throw new IllegalArgumentException(src.describe() + ": [" + from + ", +" + count + ") lies outside"
                    + " dimension " + d + " of size " + shape[d]);
        }
        if (src instanceof LoraFactorSource) {
            return LoraFactorOps.sliceDim((LoraFactorSource) src, d, from, count);
        }
        if (d == shape.length - 1) {
            return ColumnTransforms.sliceCols(src, from, count);
        }
        if (d == 0) {
            return mapDim0(src, "[" + from + ":" + (from + count) + "]", count, e -> from + e, true);
        }
        long inner = 1;
        for (int i = d + 1; i < shape.length - 1; i++) {
            inner *= shape[i];
        }
        long size = shape[d];
        long rowsPerOuterOut = count * inner;
        long[] outShape = shape.clone();
        outShape[d] = count;
        long innerRows = inner;
        RowIndex index = r -> {
            long o = r / rowsPerOuterOut;
            long w = r % rowsPerOuterOut;
            return o * size * innerRows + from * innerRows + w;
        };
        String what = src.describe() + " narrowed to [" + from + ", +" + count + ") of dim " + d;
        return new RowMap(what, Collections.singletonList(src), outShape, index, null,
                ExpertTransforms.Siblings.of(src));
    }

    /**
     * Gathers dim 0 in units of {@code unit} entries: output unit {@code u} is input unit {@code index[u]}.
     *
     * @throws IllegalArgumentException when dim 0 is not a whole number of units or an index is out of range
     */
    public static RowSource gatherRows(RowSource src, int unit, int[] index) {
        long d0 = dim0(src);
        if (unit <= 0 || d0 % unit != 0) {
            throw new IllegalArgumentException(src.describe() + ": dim 0 (" + d0 + ") is not a whole number of "
                    + unit + "-entry units");
        }
        int[] idx = index.clone();
        long units = d0 / unit;
        for (int i : idx) {
            if (i < 0 || i >= units) {
                throw new IllegalArgumentException(src.describe() + ": unit index " + i + " outside [0, " + units
                        + ")");
            }
        }
        return mapDim0(src, "gatherRows(" + unit + ")", (long) idx.length * unit,
                e -> (long) idx[(int) (e / unit)] * unit + e % unit);
    }

    // ------------------------------------------------------------------ concatenation

    /**
     * {@code torch.cat(parts, dim=0)}. Parts must agree on format and on every dimension but dim 0.
     *
     * @throws IllegalArgumentException when the parts differ in format or inner dimensions
     */
    public static RowSource concatRows(List<? extends RowSource> parts) {
        List<RowSource> ps = new ArrayList<>(Objects.requireNonNull(parts, "parts"));
        if (ps.isEmpty()) {
            throw new IllegalArgumentException("concatRows needs at least one part");
        }
        if (ps.get(0) instanceof LoraFactorSource) {
            return LoraFactorOps.concatRows(ps);
        }
        if (ps.get(0).shape().length == 1) {
            return ColumnTransforms.concatCols(ps);
        }
        ps = ExpertTransforms.Siblings.reconcile(ps, ExpertTransforms.Siblings.Merge.CONCAT_ROWS);
        RowSource first = ps.get(0);
        long[] shape = first.shape();
        long d0 = 0;
        long[] starts = new long[ps.size() + 1];
        for (int i = 0; i < ps.size(); i++) {
            RowSource p = ps.get(i);
            long[] s = p.shape();
            if (!p.format().equals(first.format()) || s.length != shape.length
                    || !Arrays.equals(Arrays.copyOfRange(s, 1, s.length), Arrays.copyOfRange(shape, 1, shape.length))) {
                throw new IllegalArgumentException("concatRows: " + p.describe() + " does not match the format and"
                        + " inner shape of " + first.describe());
            }
            d0 = Math.addExact(d0, s[0]);
            starts[i + 1] = starts[i] + p.rows();
        }
        long[] outShape = shape.clone();
        outShape[0] = d0;
        int n = ps.size();
        RowIndex index = new RowIndex() {
            @Override
            public int part(long r) {
                int lo = 0;
                int hi = n - 1;
                while (lo < hi) {
                    int mid = (lo + hi + 1) >>> 1;
                    if (starts[mid] <= r) {
                        lo = mid;
                    } else {
                        hi = mid - 1;
                    }
                }
                return lo;
            }

            @Override
            public long row(long r) {
                return r - starts[part(r)];
            }
        };
        return new RowMap("concatRows of " + n + " parts starting with " + first.describe(), ps, outShape, index,
                null, ExpertTransforms.Siblings.concatDim0(ps));
    }

    /**
     * {@code torch.cat(parts, dim)} along any dimension. Parts must agree on format and on every dimension but
     * {@code dim}; the innermost dimension is a column concatenation ({@link ColumnTransforms#concatCols}), dim 0 a
     * row concatenation ({@link #concatRows}), and a middle dimension keeps whole rows.
     *
     * @param dim the dimension, negative values counting from the innermost ({@code -1})
     * @throws IllegalArgumentException when the parts differ in format or other dimensions
     */
    public static RowSource concatDim(List<? extends RowSource> parts, int dim) {
        List<RowSource> ps = new ArrayList<>(Objects.requireNonNull(parts, "parts"));
        if (ps.isEmpty()) {
            throw new IllegalArgumentException("concatDim needs at least one part");
        }
        RowSource first = ps.get(0);
        int d = normalizeDim(first, dim);
        if (d == 0) {
            return concatRows(ps);
        }
        if (d == first.shape().length - 1) {
            return ColumnTransforms.concatCols(ps);
        }
        long[] shape = first.shape();
        long total = 0;
        long[] offsets = new long[ps.size() + 1];
        for (int i = 0; i < ps.size(); i++) {
            RowSource p = ps.get(i);
            long[] s = p.shape();
            if (!p.format().equals(first.format()) || s.length != shape.length) {
                throw new IllegalArgumentException("concatDim: " + p.describe() + " does not match the format and"
                        + " rank of " + first.describe());
            }
            for (int k = 0; k < shape.length; k++) {
                if (k != d && s[k] != shape[k]) {
                    throw new IllegalArgumentException("concatDim: " + p.describe() + " does not match dimension "
                            + k + " of " + first.describe());
                }
            }
            total = Math.addExact(total, s[d]);
            offsets[i + 1] = total;
        }
        long[] outShape = shape.clone();
        outShape[d] = total;
        // dimensions between d and the innermost one; the innermost is the row width, not a row dimension
        long tail = 1;
        for (int k = d + 1; k < shape.length - 1; k++) {
            tail = Math.multiplyExact(tail, shape[k]);
        }
        long[] sizes = new long[ps.size()];
        for (int i = 0; i < ps.size(); i++) {
            sizes[i] = ps.get(i).shape()[d];
        }
        long block = Math.multiplyExact(total, tail);
        long finalTail = tail;
        int n = ps.size();
        RowIndex index = new RowIndex() {
            @Override
            public int part(long r) {
                long j = (r % block) / finalTail;
                int lo = 0;
                int hi = n - 1;
                while (lo < hi) {
                    int mid = (lo + hi + 1) >>> 1;
                    if (offsets[mid] <= j) {
                        lo = mid;
                    } else {
                        hi = mid - 1;
                    }
                }
                return lo;
            }

            @Override
            public long row(long r) {
                long outer = r / block;
                long j = (r % block) / finalTail;
                long within = r % finalTail;
                int p = part(r);
                return (outer * sizes[p] + (j - offsets[p])) * finalTail + within;
            }
        };
        return new RowMap("concatDim of " + n + " parts along dimension " + d + " starting with "
                + first.describe(), ps, outShape, index, null, ExpertTransforms.Siblings.NONE);
    }

    // ------------------------------------------------------------------ padding and truncation

    /**
     * Pads dim 0 to {@code rows} entries, the new ones filled with {@code fill} (Gemma3N embedding pad, PocketTTS
     * extra rows). A zero fill writes zero bytes, which every ggml type decodes as 0; a non-zero fill of a block
     * source dequantizes it to F32 first.
     *
     * @throws IllegalArgumentException when {@code rows} is smaller than dim 0
     */
    public static RowSource padRows(RowSource src, long rows, float fill) {
        long d0 = dim0(src);
        if (rows < d0) {
            throw new IllegalArgumentException(src.describe() + ": cannot pad " + d0 + " entries to " + rows);
        }
        if (src instanceof LoraFactorSource) {
            if (fill != 0f) {
                throw LoraFactorOps.unsupported(src, "padRows with a non-zero fill");
            }
            return LoraFactorOps.onDim0((LoraFactorSource) src, s -> padRows(s, rows, 0f));
        }
        if (fill != 0f && src.outputScale() != null) {
            src = ElementTransforms.foldScale(src);
        }
        if (fill != 0f && !DtypePolicy.encodesConstant(src.format())) {
            src = ElementTransforms.dequantFallback(src, "padRows with fill " + fill);
        }
        if (src.shape().length == 1) {
            return ColumnTransforms.padCols(src, rows, fill);
        }
        long[] outShape = src.shape();
        outShape[0] = rows;
        long srcRows = src.rows();
        RowIndex index = new RowIndex() {
            @Override
            public int part(long r) {
                return r < srcRows ? 0 : -1;
            }

            @Override
            public long row(long r) {
                return r;
            }
        };
        byte[] fillRow = DtypePolicy.constantRow(src.format(), outShape[outShape.length - 1], fill);
        ExpertTransforms.Siblings sib = ExpertTransforms.Siblings.of(src).remapDim0(src, rows, e -> e < d0 ? e : -1);
        return new RowMap(src.describe() + " padded to " + rows + " entries with " + fill,
                Collections.singletonList(src), outShape, index, fillRow, sib);
    }

    /** The first {@code rows} entries of dim 0 (Gemma3 embedding truncation). */
    public static RowSource truncateRows(RowSource src, long rows) {
        return sliceRows(src, 0, rows);
    }

    // ------------------------------------------------------------------ shared machinery

    /** Size of dim 0. */
    static long dim0(RowSource src) {
        long[] shape = src.shape();
        if (shape.length == 0) {
            throw new IllegalArgumentException(src.describe() + " is a scalar and has no dimension 0");
        }
        return shape[0];
    }

    /** {@code dim} in {@code [0, rank)}, negative values counting from the end. */
    static int normalizeDim(RowSource src, int dim) {
        int rank = src.shape().length;
        int d = dim < 0 ? dim + rank : dim;
        if (d < 0 || d >= rank) {
            throw new IllegalArgumentException(src.describe() + " has no dimension " + dim);
        }
        return d;
    }

    /**
     * A single-source move of dim 0: output entry {@code e} is input entry {@code srcEntry(e)}. A 1-D source moves
     * elements; a contiguous slice of a tensor stored unchanged keeps its place in the checkpoint.
     */
    static RowSource mapDim0(RowSource src, String what, long outD0, LongUnaryOperator srcEntry) {
        return mapDim0(src, what, outD0, srcEntry, false);
    }

    /**
     * {@link #mapDim0(RowSource, String, long, LongUnaryOperator)}; {@code contiguous} states that
     * {@code srcEntry(e) == srcEntry(0) + e}, which keeps a tensor stored unchanged mapped in place.
     */
    static RowSource mapDim0(RowSource src, String what, long outD0, LongUnaryOperator srcEntry, boolean contiguous) {
        return mapDim0(src, what, outD0, srcEntry, contiguous, false);
    }

    /**
     * {@link #mapDim0(RowSource, String, long, LongUnaryOperator)} for a converter reorder or selection of the
     * stored rows that vLLM does not make: the result keeps them traceable through {@link #storedSource}.
     */
    static RowSource reorderDim0(RowSource src, String what, long outD0, LongUnaryOperator srcEntry) {
        return mapDim0(src, what, outD0, srcEntry, false, true);
    }

    private static RowSource mapDim0(RowSource src, String what, long outD0, LongUnaryOperator srcEntry,
                                     boolean contiguous, boolean stored) {
        if (src instanceof LoraFactorSource) {
            return LoraFactorOps.onDim0((LoraFactorSource) src,
                    s -> mapDim0(s, what, outD0, srcEntry, contiguous, stored));
        }
        long[] shape = src.shape();
        long d0 = dim0(src);
        if (shape.length == 1) {
            return ColumnTransforms.mapElements(src, what, outD0, srcEntry, contiguous);
        }
        long rowsPerEntry = d0 == 0 ? 1 : src.rows() / d0;
        long[] outShape = shape.clone();
        outShape[0] = outD0;
        RowIndex index = r -> srcEntry.applyAsLong(r / rowsPerEntry) * rowsPerEntry + r % rowsPerEntry;
        ExpertTransforms.Siblings sib = ExpertTransforms.Siblings.of(src).remapDim0(src, outD0, srcEntry);
        RowMap map = new RowMap(src.describe() + " " + what, Collections.singletonList(src), outShape, index, null,
                sib, -1, stored);
        long base = src.directVirtualOffset();
        if (contiguous && base >= 0 && outD0 > 0) {
            long offset = Math.addExact(base, Math.multiplyExact(srcEntry.applyAsLong(0) * rowsPerEntry,
                    src.rowBytes()));
            return map.withDirectOffset(offset);
        }
        return map;
    }

    /** Reads rows {@code [first, first + count)} of {@code src} into a new little-endian heap buffer. */
    static ByteBuffer readRows(RowSource src, long first, int count) throws IOException {
        long bytes = Math.multiplyExact(src.rowBytes(), (long) count);
        if (bytes > Integer.MAX_VALUE - 8) {
            throw new IllegalArgumentException(src.describe() + ": " + count + " rows of " + src.rowBytes()
                    + " bytes exceed one buffer");
        }
        ByteBuffer b = ByteBuffer.allocate((int) bytes).order(ByteOrder.LITTLE_ENDIAN);
        src.readRows(first, count, b);
        b.flip();
        return b;
    }

    /** Rows of {@code rowBytes} bytes that fit one {@link #CHUNK_BYTES} chunk; at least 1. */
    static int chunkRows(long rowBytes) {
        return (int) Math.max(1, Math.min(Integer.MAX_VALUE, CHUNK_BYTES / Math.max(1, rowBytes)));
    }

    /** Maps a flat output row to its part ({@code -1} for a fill row) and the row inside that part. */
    @FunctionalInterface
    interface RowIndex {
        long row(long r);

        default int part(long r) {
            return 0;
        }
    }

    /**
     * Base of the transform producers: an immutable shape and format with the row arithmetic and bounds checks of
     * the {@link RowSource} contract, plus the sibling values the producer carries.
     */
    abstract static class Derived implements RowSource {
        private final String describe;
        private final ElementFormat format;
        private final long[] shape;
        private final long rows;
        private final long rowBytes;
        private final ExpertTransforms.Siblings siblings;

        Derived(String describe, ElementFormat format, long[] shape, ExpertTransforms.Siblings siblings) {
            this.describe = Objects.requireNonNull(describe, "describe");
            this.format = Objects.requireNonNull(format, "format");
            this.shape = shape.clone();
            long r = 1;
            for (int i = 0; i < this.shape.length; i++) {
                if (this.shape[i] < 0) {
                    throw new IllegalArgumentException(describe + ": negative dimension in " + Arrays.toString(shape));
                }
                if (i < this.shape.length - 1) {
                    r = Math.multiplyExact(r, this.shape[i]);
                }
            }
            this.rows = r;
            long inner = this.shape.length == 0 ? 1 : this.shape[this.shape.length - 1];
            try {
                this.rowBytes = format.rowBytes(inner);
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException(describe + ": " + e.getMessage(), e);
            }
            this.siblings = siblings == null ? ExpertTransforms.Siblings.NONE : siblings;
        }

        @Override
        public final String describe() {
            return describe;
        }

        @Override
        public final ElementFormat format() {
            return format;
        }

        @Override
        public final long[] shape() {
            return shape.clone();
        }

        @Override
        public final long rows() {
            return rows;
        }

        @Override
        public final long rowBytes() {
            return rowBytes;
        }

        @Override
        public final void readRows(long first, int count, ByteBuffer dst) throws IOException {
            Objects.requireNonNull(dst, "dst");
            if (first < 0 || count < 0 || first > rows - count) {
                throw new IllegalArgumentException("rows [" + first + ", +" + count + ") lie outside the " + rows
                        + " rows of " + describe);
            }
            long bytes = Math.multiplyExact(rowBytes, (long) count);
            if (bytes > dst.remaining()) {
                throw new IllegalArgumentException(count + " rows of " + describe + " need " + bytes
                        + " bytes; the buffer has " + dst.remaining());
            }
            ByteBuffer out = dst.slice().order(ByteOrder.LITTLE_ENDIAN);
            out.limit((int) bytes);
            produce(first, count, out);
            if (out.position() != bytes) {
                throw new IllegalStateException(describe + " produced " + out.position() + " bytes for " + count
                        + " rows, expected " + bytes);
            }
            dst.position(dst.position() + (int) bytes);
        }

        /** Writes rows {@code [first, first + count)} at {@code out}'s position (little-endian, exact room). */
        abstract void produce(long first, int count, ByteBuffer out) throws IOException;

        final ExpertTransforms.Siblings siblings() {
            return siblings;
        }

        @Override
        public float[] outputScale() {
            return siblings.outputScale();
        }

        @Override
        public float[] inputScale() {
            return siblings.inputScale();
        }

        @Override
        public int[] inputZeroPoint() {
            return siblings.inputZeroPoint();
        }

        @Override
        public int[] inputPerm() {
            return siblings.inputPerm();
        }

        @Override
        public ActQuantScheme actQuant() {
            return siblings.actQuant();
        }

        @Override
        public RowSource foldInputPerm() {
            if (siblings.inputPerm() == null) {
                return RowSource.super.foldInputPerm();
            }
            return ColumnTransforms.foldPermGeneric(this);
        }

        @Override
        public RowSource foldOutputScale() {
            if (siblings.outputScale() == null) {
                return RowSource.super.foldOutputScale();
            }
            return ElementTransforms.foldScale(this);
        }

        @Override
        public String toString() {
            return getClass().getSimpleName() + "{" + describe + "}";
        }
    }

    /**
     * Rows gathered from one or more parts of the output's format and row width, by a {@link RowIndex}; runs of
     * consecutive source rows are read in one call straight into the destination. Fill rows repeat a constant row.
     */
    static final class RowMap extends Derived {
        private final List<RowSource> parts;
        private final RowIndex index;
        private final byte[] fillRow;
        private final boolean rowLocal;
        private final long directOffset;
        /** Every row reads a row of the one part, which {@link #storedSource} traces through. */
        private final boolean stored;

        RowMap(String describe, List<RowSource> parts, long[] shape, RowIndex index, byte[] fillRow,
               ExpertTransforms.Siblings siblings) {
            this(describe, parts, shape, index, fillRow, siblings, -1, false);
        }

        private RowMap(String describe, List<RowSource> parts, long[] shape, RowIndex index, byte[] fillRow,
                       ExpertTransforms.Siblings siblings, long directOffset, boolean stored) {
            super(describe, parts.get(0).format(), shape, siblings);
            this.parts = Collections.unmodifiableList(new ArrayList<>(parts));
            this.index = index;
            this.fillRow = fillRow == null ? null : fillRow.clone();
            if (this.fillRow != null && this.fillRow.length != rowBytes()) {
                throw new IllegalArgumentException(describe + ": a fill row of " + this.fillRow.length
                        + " bytes does not match rows of " + rowBytes());
            }
            boolean local = true;
            for (RowSource p : this.parts) {
                if (p.rowBytes() != rowBytes()) {
                    throw new IllegalArgumentException(describe + ": part " + p.describe() + " has rows of "
                            + p.rowBytes() + " bytes, not " + rowBytes());
                }
                local &= p.rowLocal();
            }
            this.rowLocal = local;
            this.directOffset = directOffset;
            if (stored && (this.parts.size() != 1 || this.fillRow != null)) {
                throw new IllegalArgumentException(describe + ": only a fill-free map of one part traces its stored"
                        + " rows");
            }
            this.stored = stored;
        }

        RowMap withDirectOffset(long offset) {
            return new RowMap(describe(), parts, shape(), index, fillRow, siblings(), offset, stored);
        }

        @Override
        public boolean rowLocal() {
            return rowLocal;
        }

        @Override
        public long directVirtualOffset() {
            return directOffset;
        }

        @Override
        void produce(long first, int count, ByteBuffer out) throws IOException {
            long rb = rowBytes();
            long end = first + count;
            long r = first;
            while (r < end) {
                int p = index.part(r);
                long s = index.row(r);
                int run = 1;
                while (r + run < end && index.part(r + run) == p && (p < 0 || index.row(r + run) == s + run)) {
                    run++;
                }
                if (p < 0) {
                    for (int i = 0; i < run; i++) {
                        out.put(fillRow);
                    }
                } else {
                    ByteBuffer w = out.duplicate();
                    w.limit(out.position() + (int) (rb * run));
                    parts.get(p).readRows(s, run, w);
                    out.position(w.position());
                }
                r += run;
            }
        }
    }
}
