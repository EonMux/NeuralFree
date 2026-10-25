package com.eonmux.neuralfree.gguf.hf.tensor;

import com.eonmux.neuralfree.gguf.GGMLType;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.function.LongUnaryOperator;

/**
 * Moves along the innermost dimension (PyTorch dim -1): column slices, concatenations, group permutations and
 * strided de-interleaves ({@code [..., ::2]}, the gpt-oss bias and BF16 gate/up layouts). Element formats are moved
 * byte for byte. A block format is moved whole block by whole block when every moved run starts and ends on a block
 * boundary; otherwise it is dequantized to F32 first, with one warning. All producers are row-local.
 *
 * <p>An act-order permutation is folded before columns move. Output scales are kept (each row keeps its scale);
 * activation values are kept for a single source and merged for a concatenation.</p>
 */
public final class ColumnTransforms {

    private static final Logger LOG = LogManager.getLogger(ColumnTransforms.class);

    private ColumnTransforms() {
    }

    /** {@code src[..., from : from + count]}. */
    public static RowSource sliceCols(RowSource src, long from, long count) {
        if (src instanceof LoraFactorSource) {
            return LoraFactorOps.onA((LoraFactorSource) src, a -> sliceCols(a, from, count));
        }
        long inner = inner(src);
        if (from < 0 || count < 0 || from > inner - count) {
            throw new IllegalArgumentException(src.describe() + ": columns [" + from + ", +" + count + ") lie outside"
                    + " rows of " + inner);
        }
        return single(src, "[..., " + from + ":" + (from + count) + "]", count, c -> from + c,
                src.shape().length == 1);
    }

    /**
     * {@code torch.cat(parts, dim=-1)}. Parts must agree on format and on every dimension but the innermost.
     *
     * @throws IllegalArgumentException when the parts differ in format or outer dimensions
     */
    public static RowSource concatCols(List<? extends RowSource> parts) {
        List<RowSource> ps = new ArrayList<>(Objects.requireNonNull(parts, "parts"));
        if (ps.isEmpty()) {
            throw new IllegalArgumentException("concatCols needs at least one part");
        }
        if (ps.get(0) instanceof LoraFactorSource) {
            return LoraFactorOps.concatCols(ps);
        }
        ps = ExpertTransforms.Siblings.reconcile(ps, ExpertTransforms.Siblings.Merge.CONCAT_COLS);
        RowSource first = ps.get(0);
        long[] outer = Arrays.copyOf(first.shape(), first.shape().length - 1);
        long[] starts = new long[ps.size() + 1];
        for (int i = 0; i < ps.size(); i++) {
            RowSource p = ps.get(i);
            long[] s = p.shape();
            if (s.length == 0 || !p.format().equals(first.format())
                    || !Arrays.equals(Arrays.copyOf(s, s.length - 1), outer)) {
                throw new IllegalArgumentException("concatCols: " + p.describe() + " does not match the format and"
                        + " outer shape of " + first.describe());
            }
            starts[i + 1] = Math.addExact(starts[i], s[s.length - 1]);
        }
        int n = ps.size();
        ColIndex index = new ColIndex() {
            @Override
            public int part(long c) {
                int p = Arrays.binarySearch(starts, c);
                p = p >= 0 ? p : -p - 2;
                while (p + 1 < n && starts[p + 1] == c) {
                    p++;
                }
                return p;
            }

            @Override
            public long col(long c) {
                return c - starts[part(c)];
            }
        };
        return ColMap.create("concatCols of " + n + " parts starting with " + first.describe(), ps, starts[n], index,
                0f, ExpertTransforms.Siblings.concatCols(ps), false);
    }

    /**
     * Reorders groups of {@code groupWidth} columns: output group {@code g} is input group {@code order[g]}
     * ({@code src.view(..., G, w)[..., order, :]}).
     *
     * @throws IllegalArgumentException when the rows are not whole groups or an index is out of range
     */
    public static RowSource permuteColGroups(RowSource src, int groupWidth, int[] order) {
        int[] o = order.clone();
        if (src instanceof LoraFactorSource) {
            return LoraFactorOps.onA((LoraFactorSource) src, a -> permuteColGroups(a, groupWidth, o));
        }
        long inner = inner(src);
        if (groupWidth <= 0 || inner % groupWidth != 0) {
            throw new IllegalArgumentException(src.describe() + ": rows of " + inner + " are not whole groups of "
                    + groupWidth);
        }
        long groups = inner / groupWidth;
        for (int g : o) {
            if (g < 0 || g >= groups) {
                throw new IllegalArgumentException(src.describe() + ": group index " + g + " outside [0, " + groups
                        + ")");
            }
        }
        return single(src, "column groups " + Arrays.toString(o), (long) o.length * groupWidth,
                c -> (long) o[(int) (c / groupWidth)] * groupWidth + c % groupWidth, false);
    }

    /** {@code src[..., offset::stride]}. */
    public static RowSource interleaveCols(RowSource src, int stride, int offset) {
        if (src instanceof LoraFactorSource) {
            return LoraFactorOps.onA((LoraFactorSource) src, a -> interleaveCols(a, stride, offset));
        }
        long inner = inner(src);
        if (stride <= 0 || offset < 0 || offset >= stride) {
            throw new IllegalArgumentException(src.describe() + ": [..., " + offset + "::" + stride + "] is not a"
                    + " valid stride slice");
        }
        long n = inner > offset ? (inner - offset + stride - 1) / stride : 0;
        return single(src, "[..., " + offset + "::" + stride + "]", n, c -> offset + c * stride, false);
    }

    /** {@code src[..., index]} ({@link ElementTransforms#gatherInner}). */
    static RowSource gather(RowSource src, int[] index) {
        int[] idx = index.clone();
        if (src instanceof LoraFactorSource) {
            return LoraFactorOps.onA((LoraFactorSource) src, a -> gather(a, idx));
        }
        long inner = inner(src);
        for (int i : idx) {
            if (i < 0 || i >= inner) {
                throw new IllegalArgumentException(src.describe() + ": column index " + i + " outside [0, " + inner
                        + ")");
            }
        }
        return single(src, "gathered columns", idx.length, c -> idx[(int) c], false);
    }

    /** The element move of a 1-D source behind {@link RowTransforms}' dim-0 moves. */
    static RowSource mapElements(RowSource src, String what, long n, LongUnaryOperator srcEntry, boolean contiguous) {
        return single(src, what, n, srcEntry, contiguous);
    }

    /** A 1-D source padded to {@code n} elements of {@code fill}. */
    static RowSource padCols(RowSource src, long n, float fill) {
        long inner = inner(src);
        ColIndex index = new ColIndex() {
            @Override
            public int part(long c) {
                return c < inner ? 0 : -1;
            }

            @Override
            public long col(long c) {
                return c;
            }
        };
        return ColMap.create(src.describe() + " padded to " + n + " with " + fill, Collections.singletonList(src), n,
                index, fill, ExpertTransforms.Siblings.of(src), false);
    }

    /**
     * {@code src} with its act-order permutation undone by a column gather: stored column {@code j} multiplies input
     * element {@code perm[j]}, so original column {@code perm[j]} is stored column {@code j}.
     *
     * @throws IllegalArgumentException when the permutation does not match the rows or is not a permutation
     */
    static RowSource foldPermGeneric(RowSource src) {
        int[] perm = src.inputPerm();
        if (perm == null) {
            return src;
        }
        long inner = inner(src);
        if (perm.length != inner) {
            throw new IllegalArgumentException(src.describe() + ": input permutation of " + perm.length
                    + " entries for rows of " + inner);
        }
        int[] inv = new int[perm.length];
        Arrays.fill(inv, -1);
        for (int j = 0; j < perm.length; j++) {
            if (perm[j] < 0 || perm[j] >= perm.length || inv[perm[j]] >= 0) {
                throw new IllegalArgumentException(src.describe() + ": input_perm is not a permutation of 0.."
                        + (perm.length - 1));
            }
            inv[perm[j]] = j;
        }
        ColIndex index = c -> inv[(int) c];
        return ColMap.create(src.describe() + " with its input permutation folded", Collections.singletonList(src),
                inner, index, 0f, ExpertTransforms.Siblings.of(src).withoutInputPerm(), false);
    }

    private static RowSource single(RowSource src, String what, long n, LongUnaryOperator srcCol,
                                    boolean contiguous) {
        RowSource s = ExpertTransforms.Siblings.withoutPerm(src);
        ColIndex index = srcCol::applyAsLong;
        return ColMap.create(s.describe() + " " + what, Collections.singletonList(s), n, index, 0f,
                ExpertTransforms.Siblings.of(s), contiguous);
    }

    private static long inner(RowSource src) {
        long[] shape = src.shape();
        if (shape.length == 0) {
            throw new IllegalArgumentException(src.describe() + " is a scalar and has no columns");
        }
        return shape[shape.length - 1];
    }

    /** Maps an output column to its part ({@code -1} for a fill column) and the column inside that part. */
    @FunctionalInterface
    interface ColIndex {
        long col(long c);

        default int part(long c) {
            return 0;
        }
    }

    /**
     * Output rows assembled from the same rows of one or more parts by runs of columns (elements, or whole blocks of
     * a block format). Fill columns repeat a constant element.
     */
    private static final class ColMap extends RowTransforms.Derived {
        private final List<RowSource> parts;
        /** Runs as {part, srcUnit, dstUnit, units}; part -1 is fill. */
        private final long[][] runs;
        private final int unitBytes;
        private final byte[] fillUnit;
        private final boolean rowLocal;
        private final long directOffset;

        private ColMap(String describe, List<RowSource> parts, long[] shape, long[][] runs, int unitElems,
                       byte[] fillUnit, ExpertTransforms.Siblings sib, long directOffset) {
            super(describe, parts.get(0).format(), shape, sib);
            this.parts = Collections.unmodifiableList(new ArrayList<>(parts));
            this.runs = runs;
            this.unitBytes = (int) format().rowBytes(unitElems);
            this.fillUnit = fillUnit;
            boolean local = true;
            for (RowSource p : this.parts) {
                local &= p.rowLocal();
            }
            this.rowLocal = local;
            this.directOffset = directOffset;
        }

        static RowSource create(String describe, List<RowSource> parts, long n, ColIndex index, float fill,
                                ExpertTransforms.Siblings sib, boolean contiguous) {
            RowSource p0 = parts.get(0);
            for (RowSource p : parts) {
                if (p.rows() != p0.rows()) {
                    throw new IllegalArgumentException(describe + ": " + p.describe() + " has " + p.rows()
                            + " rows, not " + p0.rows());
                }
            }
            if (n > Integer.MAX_VALUE) {
                throw new IllegalArgumentException(describe + ": rows of " + n + " elements are too long");
            }
            ElementFormat f = p0.format();
            int unit = f.isBlock() ? f.ggmlType().getBlockSize() : 1;
            boolean bytesWhole = f.isBlock() || f.stDType().bits() % 8 == 0;
            long[][] runs = bytesWhole ? runs(parts, n, index, unit, fill) : null;
            if (runs == null) {
                LOG.warn("{}: the column move splits {} {}; dequantized to F32", describe, f,
                        f.isBlock() ? "blocks" : "bytes");
                List<RowSource> fp = new ArrayList<>();
                for (RowSource p : parts) {
                    fp.add(ElementTransforms.dequant(p, GGMLType.F32));
                }
                parts = fp;
                f = fp.get(0).format();
                unit = 1;
                runs = runs(parts, n, index, 1, fill);
                sib = sib.withoutOutputScale();
            }
            long[] shape = p0.shape();
            shape[shape.length - 1] = n;
            byte[] fillUnit = DtypePolicy.constantRow(f, unit, fill);
            long direct = -1;
            if (contiguous && shape.length == 1 && runs.length == 1 && runs[0][0] == 0 && parts.get(0) == p0
                    && p0.directVirtualOffset() >= 0) {
                direct = p0.directVirtualOffset() + runs[0][1] * f.rowBytes(unit);
            }
            return new ColMap(describe, parts, shape, runs, unit, fillUnit, sib, direct);
        }

        /** Runs of whole units, or null when a unit of the output does not come from one whole source unit. */
        private static long[][] runs(List<RowSource> parts, long n, ColIndex index, int unit, float fill) {
            if (n % unit != 0) {
                return null;
            }
            List<long[]> out = new ArrayList<>();
            long[] cur = null;
            for (long u = 0; u < n / unit; u++) {
                long c0 = u * unit;
                int p = index.part(c0);
                long s0 = p < 0 ? -1 : index.col(c0);
                if (p >= 0 && s0 % unit != 0) {
                    return null;
                }
                for (int k = 1; k < unit; k++) {
                    int pk = index.part(c0 + k);
                    if (pk != p || (p >= 0 && index.col(c0 + k) != s0 + k)) {
                        return null;
                    }
                }
                if (p < 0 && unit > 1 && fill != 0f) {
                    return null;
                }
                long su = p < 0 ? -1 : s0 / unit;
                if (cur != null && cur[0] == p && (p < 0 || cur[1] + cur[3] == su)) {
                    cur[3]++;
                } else {
                    cur = new long[]{p, su, u, 1};
                    out.add(cur);
                }
            }
            for (long[] r : out) {
                if (r[0] >= 0) {
                    long inner = parts.get((int) r[0]).shape()[parts.get((int) r[0]).shape().length - 1];
                    if ((r[1] + r[3]) * unit > inner) {
                        throw new IllegalArgumentException("column run past the rows of "
                                + parts.get((int) r[0]).describe());
                    }
                }
            }
            return out.toArray(new long[0][]);
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
            long sum = rowBytes();
            for (RowSource p : parts) {
                sum += p.rowBytes();
            }
            int per = RowTransforms.chunkRows(sum);
            ByteBuffer[] in = new ByteBuffer[parts.size()];
            for (int done = 0; done < count; ) {
                int c = Math.min(per, count - done);
                for (int i = 0; i < in.length; i++) {
                    in[i] = RowTransforms.readRows(parts.get(i), first + done, c);
                }
                for (int r = 0; r < c; r++) {
                    int base = out.position();
                    for (long[] run : runs) {
                        int len = (int) (run[3] * unitBytes);
                        out.position(base + (int) (run[2] * unitBytes));
                        if (run[0] < 0) {
                            for (long k = 0; k < run[3]; k++) {
                                out.put(fillUnit);
                            }
                        } else {
                            ByteBuffer src = in[(int) run[0]].duplicate();
                            int rowStart = (int) (r * parts.get((int) run[0]).rowBytes());
                            src.position(rowStart + (int) (run[1] * unitBytes)).limit(src.position() + len);
                            out.put(src);
                        }
                    }
                    out.position(base + (int) rowBytes());
                }
                done += c;
            }
        }
    }
}
